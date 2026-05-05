"""Tiered hybrid combiner over content, collaborative, and popularity scores."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender
from src.models.popularity import PopularityRecommender
from src.features.structured_features import parse_skills
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class HybridConfig:
    warm_threshold: int = 5
    lukewarm_threshold: int = 1
    weights: dict[str, dict[str, float]] = None  # tier -> {content, collab, popularity}


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


class HybridRecommender:
    """Combine three recommenders with per-user-tier weights.
    Warm (≥ warm_threshold): content + collab. Lukewarm: heavier content. Cold (0): content profile + popularity."""

    def __init__(self, content: ContentBasedRecommender, collab: CollaborativeRecommender,
                 popularity: PopularityRecommender, cfg: HybridConfig):
        self.content = content
        self.collab = collab
        self.popularity = popularity
        self.cfg = cfg
        self._user_interaction_counts: dict[int, int] = {}
        self._user_history: dict[int, set[int]] = {}
        self._jobs: pd.DataFrame | None = None
        self._users: pd.DataFrame | None = None

    def fit(self, jobs: pd.DataFrame, users: pd.DataFrame, train: pd.DataFrame) -> "HybridRecommender":
        self._jobs = jobs
        self._users = users
        counts = train.groupby("user_id").size().to_dict()
        self._user_interaction_counts = {int(u): int(c) for u, c in counts.items()}
        hist = train.groupby("user_id")["job_id"].apply(set).to_dict()
        self._user_history = {int(u): {int(j) for j in s} for u, s in hist.items()}
        return self

    def _tier(self, user_id: int) -> str:
        c = self._user_interaction_counts.get(int(user_id), 0)
        if c >= self.cfg.warm_threshold:
            return "warm"
        if c >= self.cfg.lukewarm_threshold:
            return "lukewarm"
        return "cold"

    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = True) -> list[tuple[int, float]]:
        assert self._jobs is not None
        job_ids = self._jobs["job_id"].to_numpy()
        final = self._combined_score(user_id, list(job_ids))
        exclude = self._user_history.get(int(user_id), set()) if exclude_seen else set()
        return self._top_k(job_ids, final, k, exclude)

    # Score a candidate subset only — used by LTR's SignalProvider as a feature.
    # Returns the same per-tier weighted combination as `recommend` but on |cands|
    # jobs rather than the full corpus, so SignalProvider doesn't pay O(|all_jobs|)
    # per user during LTR fit.
    def score_pairs(self, user_id: int, job_ids: list[int]) -> np.ndarray:
        return self._combined_score(user_id, job_ids)

    # Internal: compute the tier-weighted score over an arbitrary job_id list.
    def _combined_score(self, user_id: int, job_ids: list[int]) -> np.ndarray:
        tier = self._tier(user_id)
        w = self.cfg.weights[tier]
        n = len(job_ids)
        # Cold path's content score requires the user's resume; otherwise score_pairs.
        if user_id in self.content._user_index:
            content_s = self.content.score_pairs(user_id, job_ids)
        else:
            content_s = self._content_cold_score(user_id, np.asarray(job_ids))
        if self.collab.knows_user(user_id):
            collab_s = self.collab.score_pairs(user_id, job_ids)
        else:
            collab_s = np.full(n, self.collab.global_mean, dtype=np.float32)
        pop_s = self.popularity.score_pairs(job_ids)
        return (w["content"] * _minmax(content_s)
                + w["collab"] * _minmax(collab_s)
                + w["popularity"] * _minmax(pop_s))

    # Score a brand-new user with only resume+skills (no interactions).
    def recommend_for_new_user(self, resume: str, skills: str, k: int = 10) -> list[tuple[int, float]]:
        assert self._jobs is not None
        w = self.cfg.weights["cold"]
        job_ids = self._jobs["job_id"].to_numpy()
        vec = self.content.profile_builder.build_for_new_user(resume, skills)
        # Dual profile: split [long | short] and average the two cosines.
        half = vec.shape[0] // 2
        long, short = vec[:half], vec[half:]
        je = self.content.artifacts.job_embeddings
        content_s = 0.5 * (je @ long + je @ short)
        pop_s = self.popularity.score_pairs(list(job_ids))
        final = w["content"] * _minmax(content_s) + w["popularity"] * _minmax(pop_s)
        return self._top_k(job_ids, final, k, set())

    # Provide a per-signal breakdown + overlapping skills for a single recommendation.
    def explain(self, user_id: int, job_id: int) -> dict[str, Any]:
        assert self._jobs is not None and self._users is not None
        tier = self._tier(user_id)
        w = self.cfg.weights[tier]
        content = float(self.content.score_pairs(user_id, [job_id])[0]) if user_id in self.content._user_index else 0.0
        collab = float(self.collab.predict(user_id, job_id)) if self.collab.knows_user(user_id) else self.collab.global_mean
        pop = float(self.popularity.score_pairs([job_id])[0])
        job_row = self._jobs[self._jobs["job_id"] == job_id].iloc[0]
        user_row = self._users[self._users["user_id"] == user_id].iloc[0]
        matched = sorted(parse_skills(job_row["skills"]) & parse_skills(user_row["skills"]))
        return {
            "tier": tier, "weights": dict(w),
            "scores": {"content": content, "collab": collab, "popularity": pop},
            "matched_skills": matched,
            "job_title": str(job_row["title"]),
        }

    def _content_cold_score(self, user_id: int, job_ids: np.ndarray) -> np.ndarray:
        row = self._users[self._users["user_id"] == user_id]
        if row.empty:
            return np.zeros(len(job_ids), dtype=np.float32)
        vec = self.content.profile_builder.build_for_new_user(
            str(row.iloc[0]["resume_text"]), str(row.iloc[0]["skills"]))
        half = vec.shape[0] // 2
        long, short = vec[:half], vec[half:]
        je = self.content.artifacts.job_embeddings
        return 0.5 * (je @ long + je @ short)

    def _top_k(self, job_ids: np.ndarray, scores: np.ndarray, k: int, exclude: set[int]) -> list[tuple[int, float]]:
        scores = scores.copy()
        if exclude:
            for i, j in enumerate(job_ids):
                if int(j) in exclude:
                    scores[i] = -np.inf
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(job_ids[i]), float(scores[i])) for i in top]


# Factory: build HybridConfig from Settings.models.hybrid.
def hybrid_config_from_settings(models_cfg: dict) -> HybridConfig:
    h = models_cfg["hybrid"]
    return HybridConfig(
        warm_threshold=h["warm_threshold"],
        lukewarm_threshold=h["lukewarm_threshold"],
        weights=h["weights"],
    )
