"""Shared pytest fixtures."""
from __future__ import annotations
import pandas as pd
import pytest
from pathlib import Path
import yaml
import tempfile
import shutil
from config import load_settings
from tests.fixtures.build_smoke import build as _build_smoke


# Smoke fixture: 50 users / 1000 jobs / 500 interactions, deterministic.
@pytest.fixture(scope="session")
def smoke_data_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("smoke_raw")
    _build_smoke(d)
    return d


# Small deterministic synthetic dataset for fast tests.
@pytest.fixture
def tiny_raw() -> dict[str, pd.DataFrame]:
    jobs = pd.DataFrame([
        {"job_id": 0, "title": "Backend Engineer", "category": "backend", "seniority": "mid",
         "location": "Remote", "remote": True, "skills": "python,postgresql,docker",
         "description": "Build backend services in python.", "salary_min": 120000, "salary_max": 160000, "posted_days_ago": 5},
        {"job_id": 1, "title": "Data Scientist", "category": "data_sci", "seniority": "senior",
         "location": "NYC", "remote": False, "skills": "python,pandas,pytorch",
         "description": "ML research and production models.", "salary_min": 160000, "salary_max": 220000, "posted_days_ago": 20},
        {"job_id": 2, "title": "Frontend Engineer", "category": "frontend", "seniority": "junior",
         "location": "Remote", "remote": True, "skills": "react,typescript,css",
         "description": "Build responsive UIs.", "salary_min": 90000, "salary_max": 120000, "posted_days_ago": 1},
    ])
    users = pd.DataFrame([
        {"user_id": 0, "primary_category": "backend", "seniority": "mid", "experience_years": 5,
         "preferred_location": "Remote", "skills": "python,docker",
         "resume_text": "Backend engineer with python experience."},
        {"user_id": 1, "primary_category": "frontend", "seniority": "junior", "experience_years": 2,
         "preferred_location": "Remote", "skills": "react,typescript",
         "resume_text": "Frontend developer."},
    ])
    interactions = pd.DataFrame([
        {"user_id": 0, "job_id": 0, "action": "apply", "timestamp_days_ago": 1},
        {"user_id": 0, "job_id": 1, "action": "view",  "timestamp_days_ago": 2},
        {"user_id": 0, "job_id": 2, "action": "save",  "timestamp_days_ago": 3},
        {"user_id": 1, "job_id": 2, "action": "apply", "timestamp_days_ago": 5},
        {"user_id": 1, "job_id": 0, "action": "view",  "timestamp_days_ago": 10},
    ])
    return {"jobs": jobs, "users": users, "interactions": interactions}


# Write tiny data to a temp project, return a Settings pointed at it.
@pytest.fixture
def tmp_project(tmp_path: Path, tiny_raw: dict[str, pd.DataFrame]):
    proj = tmp_path / "proj"
    (proj / "config").mkdir(parents=True)
    (proj / "data" / "raw").mkdir(parents=True)
    (proj / "data" / "processed").mkdir(parents=True)
    (proj / "models_artifacts").mkdir(parents=True)

    # Minimal config.
    cfg_src = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    with open(cfg_src) as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["use_synthetic"] = True
    cfg["data"]["synthetic"] = {"n_users": 2, "n_jobs": 3, "n_interactions": 5, "seed": 0}
    with open(proj / "config" / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    tiny_raw["jobs"].to_csv(proj / "data" / "raw" / cfg["data"]["jobs_file"], index=False)
    tiny_raw["users"].to_csv(proj / "data" / "raw" / cfg["data"]["users_file"], index=False)
    tiny_raw["interactions"].to_csv(proj / "data" / "raw" / cfg["data"]["interactions_file"], index=False)

    # Build a Settings manually pointing at this temp root.
    from config.settings import Settings, DataCfg, SyntheticCfg, SplitCfg
    settings = Settings(
        data=DataCfg(
            raw_path=cfg["data"]["raw_path"],
            processed_path=cfg["data"]["processed_path"],
            jobs_file=cfg["data"]["jobs_file"],
            users_file=cfg["data"]["users_file"],
            interactions_file=cfg["data"]["interactions_file"],
            kaggle_dataset=cfg["data"]["kaggle_dataset"],
            use_synthetic=True,
            synthetic=SyntheticCfg(**cfg["data"]["synthetic"]),
        ),
        ratings=cfg["ratings"],
        split=SplitCfg(**cfg["split"]),
        models=cfg["models"],
        faiss=cfg["faiss"],
        evaluation=cfg["evaluation"],
        paths=cfg["paths"],
        project_root=proj,
    )
    return settings
