"""Cross-feature engineering for the LTR stage.
Each row = one (user, candidate_job) pair. Features mix signals from all stage-1 models
plus engineered compatibility signals (skills, location, experience, recency)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.features.structured_features import parse_skills


# Ordered feature names — must match the booster's training vector.
# This list grew during the overhaul (added bilateral, hybrid, bert4rec). Pre-overhaul
# saved boosters are NOT compatible with this schema and must be retrained. The
# SignalProvider zero-fills any signal whose source model isn't wired in, so a
# minimally-equipped run still produces a valid 16-column feature vector.
FEATURE_NAMES: list[str] = [
    "two_tower_score",
    "content_score",
    "collab_score",
    "popularity_score",
    "skill_overlap",
    "skill_jaccard",
    "location_match",
    "experience_match",
    "salary_alignment",
    "days_since_posted",
    "user_hist_apply_rate_in_category",
    "s_user_to_job",
    "s_job_to_user",
    "bilateral_score",
    "hybrid_score",
    "bert4rec_score",
]


@dataclass
class RankingSignals:
    """Per-(user, job) signals built from stage-1 models + raw frames.

    `s_user_to_job`, `s_job_to_user`, `bilateral_score` come from the BilateralScorer.
    `hybrid` is the legacy weighted-combiner score (kept as a feature, not the final
    ranker — Task 8). `bert4rec` is the sequence model's next-item score (Task 9).
    All optional fields default to None and the feature builder zero-fills them so
    saved boosters with prior columns can still be loaded by trimming the trailing
    columns.
    """
    two_tower: np.ndarray       # (n_pairs,)
    content: np.ndarray
    collab: np.ndarray
    popularity: np.ndarray
    s_user_to_job: np.ndarray | None = None
    s_job_to_user: np.ndarray | None = None
    bilateral: np.ndarray | None = None
    hybrid: np.ndarray | None = None
    bert4rec: np.ndarray | None = None


def _experience_match(user_years: int, job_seniority: str) -> float:
    ranges = {"junior": (0, 2), "mid": (2, 5), "senior": (5, 10), "staff": (10, 20), "principal": (15, 40)}
    lo, hi = ranges.get(str(job_seniority).lower(), (0, 40))
    if lo <= user_years <= hi:
        return 1.0
    return max(0.0, 1.0 - min(abs(user_years - lo), abs(user_years - hi)) / 10.0)


def _salary_alignment(user_years: int, sal_min: float, sal_max: float,
                      salary_model=None, job_features: dict | None = None) -> float:
    """Alignment between user's expected comp and the job's salary band.

    When `salary_model` and `job_features` are provided, prefer the learned salary
    band (Indeed/LinkedIn approach). Falls back to the posted band; if that's also
    missing, falls back to the original "70k + 12k*years" heuristic.
    """
    expected = 70_000 + 12_000 * user_years
    if salary_model is not None and job_features is not None:
        try:
            pred = salary_model.predict(**job_features)
            sal_min = float(pred.salary_min) or sal_min
            sal_max = float(pred.salary_max) or sal_max
        except Exception:
            pass
    if not np.isfinite(sal_min) or not np.isfinite(sal_max) or sal_max <= 0:
        return 0.0
    if sal_min <= expected <= sal_max:
        return 1.0
    gap = min(abs(expected - sal_min), abs(expected - sal_max))
    return max(0.0, 1.0 - gap / max(sal_max - sal_min, 1.0))


# User's historical apply rate per job category (from train interactions).
def user_category_apply_rates(train: pd.DataFrame, jobs: pd.DataFrame,
                              apply_rating: int = 5) -> dict[tuple[int, str], float]:
    if "category" not in jobs.columns:
        return {}
    j2c = dict(zip(jobs["job_id"], jobs["category"]))
    df = train.copy()
    df["category"] = df["job_id"].map(j2c)
    out: dict[tuple[int, str], float] = {}
    for (uid, cat), g in df.groupby(["user_id", "category"]):
        out[(int(uid), str(cat))] = float((g["rating"] >= apply_rating).mean())
    return out


# Build the (n_pairs, n_features) matrix for a list of user-job candidates.
# `jobs` may be passed either as the raw frame or pre-indexed by job_id — accepting
# both lets hot-path callers (LTR fit) hoist the set_index call out of the per-user
# loop, saving ~120ms × n_users on the real-scale dataset (~5 min for 2,484 users).
def build_ranking_features(
    user_id: int, candidate_job_ids: list[int],
    users: pd.DataFrame, jobs: pd.DataFrame,
    signals: RankingSignals,
    user_cat_apply_rates: dict[tuple[int, str], float] | None = None,
    salary_model=None,
) -> np.ndarray:
    user_row = users[users["user_id"] == user_id]
    if user_row.empty:
        user_years = 0
        user_skills = ""
        user_loc = ""
    else:
        user_years = int(user_row.iloc[0].get("experience_years", 0))
        user_skills = str(user_row.iloc[0].get("skills", ""))
        user_loc = str(user_row.iloc[0].get("preferred_location", ""))

    job_by_id = jobs if jobs.index.name == "job_id" else jobs.set_index("job_id")
    n = len(candidate_job_ids)
    feats = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)

    u_skill_set = parse_skills(user_skills)
    for i, jid in enumerate(candidate_job_ids):
        j = job_by_id.loc[jid] if jid in job_by_id.index else None
        j_skills = str(j["skills"]) if j is not None else ""
        j_skill_set = parse_skills(j_skills)
        overlap = len(u_skill_set & j_skill_set)
        union = max(len(u_skill_set | j_skill_set), 1)
        jaccard = overlap / union

        loc_match = 1.0 if j is not None and str(j["location"]).lower() == user_loc.lower() else 0.0
        exp_match = _experience_match(user_years, str(j.get("seniority", ""))) if j is not None else 0.0
        if j is not None:
            jf = ({"category": str(j.get("category", "")),
                   "seniority": str(j.get("seniority", "")),
                   "location": str(j.get("location", "")),
                   "skills": j_skills} if salary_model is not None else None)
            sal_align = _salary_alignment(
                user_years, float(j.get("salary_min", 0) or 0),
                float(j.get("salary_max", 0) or 0),
                salary_model=salary_model, job_features=jf,
            )
        else:
            sal_align = 0.0
        posted = float(j.get("posted_days_ago", 0) or 0) if j is not None else 0.0
        cat = str(j.get("category", "")) if j is not None else ""
        apply_rate = (user_cat_apply_rates or {}).get((int(user_id), cat), 0.0)

        s_uj = float(signals.s_user_to_job[i]) if signals.s_user_to_job is not None else 0.0
        s_ju = float(signals.s_job_to_user[i]) if signals.s_job_to_user is not None else 0.0
        bilat = float(signals.bilateral[i]) if signals.bilateral is not None else 0.0
        hyb = float(signals.hybrid[i]) if signals.hybrid is not None else 0.0
        b4r = float(signals.bert4rec[i]) if signals.bert4rec is not None else 0.0

        feats[i] = [
            float(signals.two_tower[i]),
            float(signals.content[i]),
            float(signals.collab[i]),
            float(signals.popularity[i]),
            float(overlap),
            float(jaccard),
            loc_match,
            exp_match,
            sal_align,
            posted,
            float(apply_rate),
            s_uj, s_ju, bilat,
            hyb, b4r,
        ]
    return feats
