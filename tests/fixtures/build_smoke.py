"""Deterministic small dataset for fast end-to-end smoke tests.
50 users, 1000 jobs, ~500 interactions. Same canonical schema as production."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

SKILLS = ["python", "sql", "java", "react", "aws", "ml", "go", "rust", "tf", "kubernetes"]
CATS = ["software", "data", "devops", "frontend", "ml"]
LOCS = ["NYC, NY, USA", "SF, CA, USA", "Remote, , ", "Berlin, , Germany",
        "London, , UK", "Austin, TX, USA"]
SEN = ["junior", "mid", "senior", "staff"]


def build(out_dir: Path, n_users: int = 50, n_jobs: int = 1000,
          n_interactions: int = 500, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    cats = rng.choice(CATS, n_jobs)
    jobs = pd.DataFrame({
        "job_id": np.arange(n_jobs),
        "title": [f"{c}_engineer_{i}" for i, c in enumerate(cats)],
        "description": [f"role description {i}" for i in range(n_jobs)],
        "category": cats,
        "seniority": rng.choice(SEN, n_jobs),
        "location": rng.choice(LOCS, n_jobs),
        "skills": [",".join(rng.choice(SKILLS, rng.integers(2, 5), replace=False))
                   for _ in range(n_jobs)],
        "salary_min": rng.integers(60_000, 120_000, n_jobs),
        "salary_max": rng.integers(120_000, 200_000, n_jobs),
        "posted_days_ago": rng.integers(0, 60, n_jobs),
    })
    users = pd.DataFrame({
        "user_id": np.arange(n_users),
        "resume_text": [f"user {i} resume" for i in range(n_users)],
        "skills": [",".join(rng.choice(SKILLS, rng.integers(2, 5), replace=False))
                   for _ in range(n_users)],
        "experience_years": rng.integers(0, 15, n_users),
        "preferred_location": rng.choice(LOCS, n_users),
    })
    inter = pd.DataFrame({
        "user_id": rng.integers(0, n_users, n_interactions),
        "job_id": rng.integers(0, n_jobs, n_interactions),
        "action": rng.choice(["view", "save", "apply"], n_interactions, p=[0.7, 0.2, 0.1]),
        "timestamp_days_ago": rng.integers(0, 90, n_interactions),
    })
    jobs.to_csv(out_dir / "jobs.csv", index=False)
    users.to_csv(out_dir / "users.csv", index=False)
    inter.to_csv(out_dir / "interactions.csv", index=False)


if __name__ == "__main__":
    out = Path("data/smoke")
    build(out)
    print(f"Built smoke dataset at {out}")
