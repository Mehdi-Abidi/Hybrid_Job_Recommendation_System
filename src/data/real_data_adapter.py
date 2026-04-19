"""Adapters that turn raw Kaggle datasets into the canonical 3-CSV schema.

Sources:
  - data/raw/postings.csv     (LinkedIn jobs, ~33K)          -> jobs.csv
  - data/raw/Resume.csv       (labeled resumes, ~2.5K)       -> users.csv
  - data/raw/Train_rev1.csv   (UK jobs w/ salaries, ~240K)   -> salary training aux

Interactions are synthesized against real jobs with category/skill-correlated signal
so collaborative + Two-Tower models learn from realistic noise patterns.

Canonical schema:
  jobs.csv          job_id, title, category, seniority, location, skills, description,
                    salary_min, salary_max, posted_days_ago
  users.csv         user_id, primary_category, seniority, experience_years,
                    preferred_location, skills, resume_text
  interactions.csv  user_id, job_id, action, timestamp_days_ago
"""
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd

from src.nlp.resume_ner import ResumeNER
from src.ontology.skill_ontology import SkillOntology
from src.utils.logging import get_logger

log = get_logger(__name__)


# Broad Resume.csv categories -> internal primary_category buckets used everywhere.
_RESUME_CAT_MAP: dict[str, str] = {
    "INFORMATION-TECHNOLOGY": "backend", "ENGINEERING": "backend",
    "DIGITAL-MEDIA": "frontend", "DESIGNER": "frontend",
    "BUSINESS-DEVELOPMENT": "data_sci", "CONSULTANT": "data_sci",
    "FINANCE": "data_sci", "ACCOUNTANT": "data_sci", "BANKING": "data_sci",
    "SALES": "data_sci", "HR": "data_sci", "PUBLIC-RELATIONS": "data_sci",
    "HEALTHCARE": "data_sci", "FITNESS": "data_sci", "CHEF": "data_sci",
    "TEACHER": "data_sci", "AVIATION": "devops", "CONSTRUCTION": "devops",
    "AUTOMOBILE": "devops", "AGRICULTURE": "devops", "APPAREL": "frontend",
    "ARTS": "frontend", "ADVOCATE": "data_sci", "BPO": "data_sci",
}

_SENIORITY_MAP = {
    "MID_SENIOR_LEVEL": "senior", "SENIOR_LEVEL": "senior",
    "DIRECTOR": "principal", "EXECUTIVE": "principal",
    "ASSOCIATE": "mid", "ENTRY_LEVEL": "junior", "INTERNSHIP": "junior",
}


def _infer_category_from_title(title: str, ontology: SkillOntology) -> str:
    t = title.lower()
    # Explicit keyword rules beat ontology vote for job titles since titles are terse.
    if any(k in t for k in ("frontend", "front-end", "front end", "react", "ui ", "ux ")):
        return "frontend"
    if any(k in t for k in ("backend", "back-end", "back end", "server", "api ")):
        return "backend"
    if any(k in t for k in ("data scien", "ml ", "machine learn", "ai ", "analytics", "analyst")):
        return "data_sci"
    if any(k in t for k in ("devops", "sre", "platform", "infra", "cloud")):
        return "devops"
    if any(k in t for k in ("mobile", "ios", "android")):
        return "mobile"
    if any(k in t for k in ("security", "pentest", "infosec")):
        return "security"
    return "backend"  # fallback: dominant category in tech postings


def _infer_seniority_from_title(title: str, exp_level: str) -> str:
    mapped = _SENIORITY_MAP.get(str(exp_level).upper().replace(" ", "_"), "")
    if mapped:
        return mapped
    t = title.lower()
    if "senior" in t or "sr." in t or "lead" in t:
        return "senior"
    if "staff" in t or "principal" in t:
        return "principal"
    if "junior" in t or "jr." in t or "intern" in t:
        return "junior"
    return "mid"


def _extract_skills(text: str, ontology: SkillOntology, limit: int = 8) -> list[str]:
    t = str(text).lower()
    found: list[str] = []
    seen: set[str] = set()
    skill_pool = sorted(ontology.all_skills()) if hasattr(ontology, "all_skills") else _all_skills(ontology)
    for skill in skill_pool:
        if skill in seen:
            continue
        if re.search(rf"\b{re.escape(skill)}\b", t):
            found.append(skill); seen.add(skill)
            if len(found) >= limit:
                break
    return found


def _all_skills(ontology: SkillOntology) -> list[str]:
    # Flatten the seed skills dict; tolerant to ontology API variants.
    out: set[str] = set()
    for skills in ontology.categories.values():
        out.update(ontology.normalize(s) for s in skills)
    return sorted(out)


def _strip_html(text: str, max_chars: int = 1200) -> str:
    t = re.sub(r"<[^>]+>", " ", str(text))
    t = re.sub(r"\s+", " ", t).strip()
    return t[:max_chars]


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

def adapt_postings(postings_path: Path, ontology: SkillOntology,
                   max_jobs: int | None = None) -> pd.DataFrame:
    """LinkedIn postings.csv -> canonical jobs.csv schema."""
    df = pd.read_csv(postings_path, low_memory=False)
    if max_jobs:
        df = df.head(max_jobs)
    log.info("Adapting %d LinkedIn postings", len(df))

    out = pd.DataFrame()
    out["job_id"] = df["job_id"].astype("int64")
    out["title"] = df["title"].fillna("").astype(str).str.strip()
    out["description"] = df["description"].fillna(out["title"]).astype(str).apply(_strip_html)
    out["location"] = df["location"].fillna("Unknown").astype(str).str.strip()
    out["category"] = out["title"].apply(lambda t: _infer_category_from_title(t, ontology))
    out["seniority"] = [_infer_seniority_from_title(t, lvl)
                        for t, lvl in zip(out["title"], df.get("formatted_experience_level", ""))]

    # Skills pulled from title + description + skills_desc via ontology match.
    corpus = (out["title"] + " " + df.get("skills_desc", "").fillna("") + " " + out["description"])
    out["skills"] = corpus.apply(lambda x: ",".join(_extract_skills(x, ontology)))

    # Salary from normalized_salary if present, else min/max, else category default.
    smin = pd.to_numeric(df.get("min_salary"), errors="coerce")
    smax = pd.to_numeric(df.get("max_salary"), errors="coerce")
    norm = pd.to_numeric(df.get("normalized_salary"), errors="coerce")
    out["salary_min"] = smin.fillna(norm * 0.85).fillna(70000).clip(lower=30000, upper=400000).astype(int)
    out["salary_max"] = smax.fillna(norm * 1.15).fillna(110000).clip(lower=40000, upper=600000).astype(int)
    out.loc[out["salary_max"] < out["salary_min"], "salary_max"] = out["salary_min"] + 20000

    # Recency: listed_time is a unix ms timestamp; convert to days-ago (fallback 30).
    listed = pd.to_numeric(df.get("listed_time"), errors="coerce")
    now_ms = listed.max() if listed.notna().any() else None
    if now_ms:
        out["posted_days_ago"] = ((now_ms - listed) / (1000 * 86400)).fillna(30).clip(lower=0, upper=365).astype(int)
    else:
        out["posted_days_ago"] = 30

    out = out.drop_duplicates(subset=["job_id"])
    out = out[out["title"] != ""].reset_index(drop=True)
    return out[["job_id", "title", "category", "seniority", "location",
                "skills", "description", "salary_min", "salary_max", "posted_days_ago"]]


def adapt_resumes(resumes_path: Path, ontology: SkillOntology,
                  jobs: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Resume.csv -> canonical users.csv schema. One user per resume.
    Uses ResumeNER (spaCy if installed, regex otherwise) for richer skill/experience extraction."""
    df = pd.read_csv(resumes_path, low_memory=False)
    log.info("Adapting %d resumes", len(df))
    rng = np.random.default_rng(seed)
    ner = ResumeNER(skill_vocab=ontology.all_skills(), use_spacy=True)

    locations = jobs["location"].value_counts().head(20).index.tolist() or ["Remote"]

    out = pd.DataFrame()
    out["user_id"] = np.arange(len(df), dtype=np.int64)
    out["resume_text"] = df["Resume_str"].fillna("").astype(str).apply(lambda s: _strip_html(s, 2000))
    out["primary_category"] = df["Category"].fillna("").astype(str).str.upper().map(_RESUME_CAT_MAP).fillna("backend")

    # Seniority inferred from resume length + "senior"/"lead" keywords.
    text_lower = out["resume_text"].str.lower()
    out["seniority"] = np.where(
        text_lower.str.contains(r"\b(?:principal|director|vp|head of)\b", regex=True), "principal",
        np.where(text_lower.str.contains(r"\b(?:senior|sr\.|lead|staff)\b", regex=True), "senior",
        np.where(text_lower.str.contains(r"\b(?:junior|jr\.|intern|entry)\b", regex=True), "junior", "mid")),
    )

    # Extract entities via NER (experience years + skills); fall back to heuristics per row.
    ner_out = [ner.parse(t) for t in out["resume_text"]]
    lengths = out["resume_text"].str.len().to_numpy()
    out["experience_years"] = [
        e.experience_years if e.experience_years > 0 else int(np.clip(l // 250, 1, 25))
        for e, l in zip(ner_out, lengths)
    ]
    out["preferred_location"] = rng.choice(locations, size=len(df))
    out["skills"] = [
        ",".join(sorted(set(e.skills) | set(_extract_skills(t, ontology))))
        for e, t in zip(ner_out, out["resume_text"])
    ]

    return out[["user_id", "primary_category", "seniority", "experience_years",
                "preferred_location", "skills", "resume_text"]]


def synthesize_interactions(users: pd.DataFrame, jobs: pd.DataFrame,
                            avg_per_user: int = 12, seed: int = 42) -> pd.DataFrame:
    """Generate (user, job, action, timestamp_days_ago) with category/skill-correlated signal.

    Matching category boosts prob; each matching skill adds smaller boost. Action tier
    (view/save/apply) depends on match strength so stronger signals become stronger ratings."""
    rng = np.random.default_rng(seed)
    n_users, n_jobs = len(users), len(jobs)
    log.info("Synthesizing interactions over %d users x %d jobs", n_users, n_jobs)

    job_cats = jobs["category"].to_numpy()
    job_skills = [set(s.split(",")) if s else set() for s in jobs["skills"].fillna("")]
    job_ids = jobs["job_id"].to_numpy()

    rows = []
    for _, u in users.iterrows():
        u_cat = u["primary_category"]
        u_skills = set(str(u["skills"]).split(",")) if u["skills"] else set()

        # Score every job by category match (1.0) + skill overlap (0.3 each).
        cat_match = (job_cats == u_cat).astype(float)
        skill_overlap = np.array([len(u_skills & js) for js in job_skills], dtype=float)
        scores = cat_match + 0.3 * skill_overlap + rng.normal(0, 0.15, n_jobs)

        # Sample top candidates with softmax over scores; number per user varies.
        n = max(3, int(rng.poisson(avg_per_user)))
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()
        picks = rng.choice(n_jobs, size=min(n, n_jobs), replace=False, p=probs)

        for ji in picks:
            s = scores[ji]
            # Action tier by score percentile: higher -> apply, mid -> save, low -> view.
            if s > 1.2:
                action = "apply"
            elif s > 0.5:
                action = "save"
            else:
                action = "view"
            rows.append((int(u["user_id"]), int(job_ids[ji]), action, int(rng.integers(1, 180))))

    df = pd.DataFrame(rows, columns=["user_id", "job_id", "action", "timestamp_days_ago"])
    log.info("Generated %d interactions (%.1f/user avg)", len(df), len(df) / max(1, n_users))
    return df


def build_real_dataset(raw_dir: Path, out_dir: Path, ontology: SkillOntology | None = None,
                       max_jobs: int | None = None, seed: int = 42) -> dict[str, Path]:
    """Full pipeline: real CSVs -> canonical triple written to `out_dir`."""
    ontology = ontology or SkillOntology()
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = adapt_postings(raw_dir / "postings.csv", ontology, max_jobs=max_jobs)
    users = adapt_resumes(raw_dir / "Resume.csv", ontology, jobs, seed=seed)
    interactions = synthesize_interactions(users, jobs, seed=seed)

    paths = {
        "jobs": out_dir / "jobs.csv",
        "users": out_dir / "users.csv",
        "interactions": out_dir / "interactions.csv",
    }
    jobs.to_csv(paths["jobs"], index=False)
    users.to_csv(paths["users"], index=False)
    interactions.to_csv(paths["interactions"], index=False)
    log.info("Real dataset built: %d jobs, %d users, %d interactions", len(jobs), len(users), len(interactions))
    return paths
