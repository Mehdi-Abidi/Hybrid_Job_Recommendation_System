"""Dataset acquisition: Kaggle download or synthetic generation."""
from __future__ import annotations
import os
import json
import zipfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

from config import Settings, load_settings
from src.utils.logging import get_logger

log = get_logger(__name__)


# Realistic skill pool grouped by job family — generator correlates skills to categories.
SKILL_POOL: dict[str, list[str]] = {
    "backend":    ["python", "java", "go", "node.js", "postgresql", "redis", "docker", "kubernetes", "rest api", "grpc", "microservices", "kafka", "rabbitmq"],
    "frontend":   ["javascript", "typescript", "react", "vue", "angular", "html", "css", "tailwind", "webpack", "next.js", "redux"],
    "data_sci":   ["python", "pandas", "numpy", "scikit-learn", "pytorch", "tensorflow", "sql", "statistics", "nlp", "computer vision", "xgboost", "spark"],
    "data_eng":   ["python", "sql", "airflow", "spark", "kafka", "dbt", "snowflake", "bigquery", "aws", "etl", "data warehouse"],
    "devops":     ["aws", "gcp", "azure", "terraform", "kubernetes", "docker", "jenkins", "ansible", "prometheus", "grafana", "linux", "bash"],
    "mobile":     ["swift", "kotlin", "react native", "flutter", "ios", "android", "objective-c", "dart"],
    "ml_eng":     ["python", "pytorch", "tensorflow", "mlflow", "kubeflow", "docker", "aws", "sagemaker", "huggingface", "llms", "vector databases"],
    "security":   ["penetration testing", "siem", "owasp", "cryptography", "aws security", "zero trust", "soc", "python", "incident response"],
    "product":    ["roadmapping", "user research", "jira", "sql", "a/b testing", "analytics", "figma"],
    "design":     ["figma", "sketch", "adobe xd", "user research", "prototyping", "accessibility", "design systems"],
}

CATEGORY_TITLES: dict[str, list[str]] = {
    "backend":  ["Backend Engineer", "Senior Backend Engineer", "Staff Backend Engineer", "Software Engineer (Backend)"],
    "frontend": ["Frontend Engineer", "Senior Frontend Engineer", "UI Engineer", "Web Developer"],
    "data_sci": ["Data Scientist", "Senior Data Scientist", "Applied Scientist", "Research Scientist"],
    "data_eng": ["Data Engineer", "Senior Data Engineer", "Analytics Engineer"],
    "devops":   ["DevOps Engineer", "Site Reliability Engineer", "Platform Engineer", "Cloud Engineer"],
    "mobile":   ["iOS Engineer", "Android Engineer", "Mobile Developer"],
    "ml_eng":   ["Machine Learning Engineer", "Senior ML Engineer", "MLOps Engineer"],
    "security": ["Security Engineer", "Application Security Engineer", "Security Analyst"],
    "product":  ["Product Manager", "Senior Product Manager", "Technical Product Manager"],
    "design":   ["Product Designer", "UX Designer", "Design Lead"],
}

LOCATIONS = [
    "San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX", "Boston, MA",
    "Chicago, IL", "Los Angeles, CA", "Denver, CO", "Atlanta, GA", "Remote",
    "London, UK", "Berlin, DE", "Toronto, CA", "Amsterdam, NL", "Singapore",
]

SENIORITY_LEVELS = ["entry", "junior", "mid", "senior", "staff"]
EXPERIENCE_BY_LEVEL = {"entry": (0, 2), "junior": (2, 4), "mid": (4, 7), "senior": (7, 12), "staff": (12, 20)}
SALARY_BY_LEVEL = {"entry": (60, 95), "junior": (85, 120), "mid": (115, 160), "senior": (155, 230), "staff": (210, 340)}  # k USD


# Deterministic RNG wrapper.
class _Rng:
    def __init__(self, seed: int):
        self.np = np.random.default_rng(seed)

    def choice(self, seq, size=None, replace=True, p=None):
        return self.np.choice(seq, size=size, replace=replace, p=p)

    def sample_skills(self, pool: list[str], k_min: int, k_max: int) -> list[str]:
        k = int(self.np.integers(k_min, k_max + 1))
        k = min(k, len(pool))
        idx = self.np.choice(len(pool), size=k, replace=False)
        return [pool[i] for i in idx]


# Generate synthetic jobs, users, interactions with realistic category-skill correlations.
def generate_synthetic(cfg: Settings) -> dict[str, pd.DataFrame]:
    s = cfg.data.synthetic
    rng = _Rng(s.seed)
    cats = list(SKILL_POOL.keys())

    # jobs
    rows = []
    for jid in range(s.n_jobs):
        cat = str(rng.choice(cats))
        level = str(rng.choice(SENIORITY_LEVELS, p=[0.10, 0.20, 0.35, 0.25, 0.10]))
        title = f"{'Senior ' if level in ('senior', 'staff') and 'Senior' not in str(rng.choice(CATEGORY_TITLES[cat])) else ''}{rng.choice(CATEGORY_TITLES[cat])}"
        required = rng.sample_skills(SKILL_POOL[cat], 3, 7)
        salary_lo, salary_hi = SALARY_BY_LEVEL[level]
        s_lo = int(rng.np.integers(salary_lo, salary_lo + 20)) * 1000
        s_hi = int(rng.np.integers(salary_hi - 15, salary_hi + 5)) * 1000
        posted_days_ago = int(rng.np.integers(0, 90))
        rows.append({
            "job_id": jid,
            "title": title,
            "category": cat,
            "seniority": level,
            "location": str(rng.choice(LOCATIONS)),
            "remote": bool(rng.np.random() < 0.35),
            "skills": ",".join(required),
            "description": _job_description(title, cat, required, level),
            "salary_min": s_lo,
            "salary_max": s_hi,
            "posted_days_ago": posted_days_ago,
        })
    jobs = pd.DataFrame(rows)

    # users
    user_rows = []
    for uid in range(s.n_users):
        primary = str(rng.choice(cats))
        # 30% of users have overlap with a secondary category
        secondary = str(rng.choice(cats)) if rng.np.random() < 0.3 else None
        pool = SKILL_POOL[primary] + (SKILL_POOL[secondary] if secondary else [])
        pool = list(dict.fromkeys(pool))
        skills = rng.sample_skills(pool, 3, 8)
        level = str(rng.choice(SENIORITY_LEVELS, p=[0.15, 0.25, 0.30, 0.20, 0.10]))
        exp_lo, exp_hi = EXPERIENCE_BY_LEVEL[level]
        user_rows.append({
            "user_id": uid,
            "primary_category": primary,
            "seniority": level,
            "experience_years": int(rng.np.integers(exp_lo, exp_hi + 1)),
            "preferred_location": str(rng.choice(LOCATIONS)),
            "skills": ",".join(skills),
            "resume_text": _resume_text(skills, level, primary),
        })
    users = pd.DataFrame(user_rows)

    # interactions — biased by skill overlap + category match + seniority fit
    interactions = _generate_interactions(users, jobs, s.n_interactions, rng)

    return {"jobs": jobs, "users": users, "interactions": interactions}


# Build a job description stitching required skills into a template.
def _job_description(title: str, category: str, skills: list[str], level: str) -> str:
    skill_clause = ", ".join(skills[:-1]) + f", and {skills[-1]}" if len(skills) > 1 else skills[0]
    return (
        f"We are hiring a {title} to join our {category.replace('_', ' ')} team. "
        f"You will work with {skill_clause}. "
        f"The ideal candidate is {level}-level with strong problem-solving skills and a track record of shipping production systems."
    )


# Build a short resume-style paragraph for each user.
def _resume_text(skills: list[str], level: str, category: str) -> str:
    return (
        f"{level.capitalize()} engineer specializing in {category.replace('_', ' ')}. "
        f"Proficient in {', '.join(skills)}. Experienced in shipping production-grade software."
    )


# Correlate interactions with skill/category/seniority fit so models have learnable signal.
def _generate_interactions(users: pd.DataFrame, jobs: pd.DataFrame, n_interactions: int, rng: _Rng) -> pd.DataFrame:
    user_skill_sets = {r.user_id: set(r.skills.split(",")) for r in users.itertuples()}
    user_cat = users.set_index("user_id")["primary_category"].to_dict()
    user_level = users.set_index("user_id")["seniority"].to_dict()
    job_skill_sets = {r.job_id: set(r.skills.split(",")) for r in jobs.itertuples()}
    job_cat = jobs.set_index("job_id")["category"].to_dict()
    job_level = jobs.set_index("job_id")["seniority"].to_dict()

    level_idx = {lvl: i for i, lvl in enumerate(SENIORITY_LEVELS)}
    action_types = ["view", "save", "apply"]
    action_probs_strong = [0.40, 0.25, 0.35]  # good match: more saves/applies
    action_probs_weak   = [0.85, 0.10, 0.05]  # weak match: mostly views

    rows = []
    interactions_per_user = max(1, n_interactions // len(users))
    for uid in users["user_id"]:
        u_skills = user_skill_sets[uid]
        u_cat = user_cat[uid]
        u_lvl = level_idx[user_level[uid]]

        # Score a candidate pool (all jobs) by overlap + category match + seniority proximity
        job_ids = jobs["job_id"].to_numpy()
        scores = np.zeros(len(job_ids), dtype=np.float64)
        for i, jid in enumerate(job_ids):
            overlap = len(u_skills & job_skill_sets[jid]) / max(1, len(job_skill_sets[jid]))
            cat_bonus = 0.5 if job_cat[jid] == u_cat else 0.0
            lvl_penalty = -0.15 * abs(u_lvl - level_idx[job_level[jid]])
            scores[i] = overlap + cat_bonus + lvl_penalty

        # Softmax → sampling distribution (temperature smooths)
        temp = 0.6
        probs = np.exp(scores / temp)
        probs = probs / probs.sum()

        n = int(np.clip(rng.np.normal(interactions_per_user, interactions_per_user * 0.3), 1, 100))
        picked = rng.np.choice(job_ids, size=min(n, len(job_ids)), replace=False, p=probs)
        for jid in picked:
            overlap = len(u_skills & job_skill_sets[int(jid)]) / max(1, len(job_skill_sets[int(jid)]))
            cat_match = job_cat[int(jid)] == u_cat
            strong = overlap >= 0.3 or cat_match
            action = str(rng.np.choice(action_types, p=action_probs_strong if strong else action_probs_weak))
            rows.append({
                "user_id": int(uid),
                "job_id": int(jid),
                "action": action,
                "timestamp_days_ago": int(rng.np.integers(0, 180)),
            })

    return pd.DataFrame(rows).drop_duplicates(["user_id", "job_id"]).reset_index(drop=True)


# Attempt Kaggle download; raise if credentials or package missing.
def download_kaggle(cfg: Settings) -> dict[str, pd.DataFrame]:
    raw_dir = cfg.path("raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise RuntimeError("kaggle package not installed. pip install kaggle or set use_synthetic=true.") from e
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(cfg.data.kaggle_dataset, path=str(raw_dir), unzip=True)
    log.info("Kaggle download complete at %s", raw_dir)
    # NOTE: downstream parsing depends on the specific dataset schema; stub for now.
    raise NotImplementedError(
        "Kaggle dataset parsing not implemented. Supply a custom loader or use synthetic data."
    )


# Persist the three dataframes to raw/ as CSV.
def save_raw(datasets: dict[str, pd.DataFrame], cfg: Settings) -> None:
    raw = cfg.path("raw")
    raw.mkdir(parents=True, exist_ok=True)
    datasets["jobs"].to_csv(raw / cfg.data.jobs_file, index=False)
    datasets["users"].to_csv(raw / cfg.data.users_file, index=False)
    datasets["interactions"].to_csv(raw / cfg.data.interactions_file, index=False)
    log.info("Saved raw dataset: %d jobs, %d users, %d interactions",
             len(datasets["jobs"]), len(datasets["users"]), len(datasets["interactions"]))


# Main entrypoint: synth or kaggle based on config, then save.
def main() -> None:
    cfg = load_settings()
    if cfg.data.use_synthetic:
        log.info("Generating synthetic dataset (n_users=%d, n_jobs=%d, n_interactions=%d)",
                 cfg.data.synthetic.n_users, cfg.data.synthetic.n_jobs, cfg.data.synthetic.n_interactions)
        datasets = generate_synthetic(cfg)
    else:
        datasets = download_kaggle(cfg)
    save_raw(datasets, cfg)


if __name__ == "__main__":
    main()
