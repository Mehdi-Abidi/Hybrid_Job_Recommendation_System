"""Adapter produces canonical-schema output from tiny fixture CSVs."""
from __future__ import annotations
import pandas as pd

from src.data.real_data_adapter import build_real_dataset
from src.ontology.skill_ontology import SkillOntology


def _write_fixtures(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    postings = pd.DataFrame({
        "job_id": [1, 2, 3],
        "title": ["Senior Python Backend Engineer", "React Frontend Developer", "Data Scientist"],
        "description": ["<p>Build APIs with python, django, postgresql.</p>",
                        "React, typescript, tailwind.",
                        "pytorch, pandas, sklearn."],
        "location": ["New York", "Remote", "SF"],
        "min_salary": [140000, 90000, 160000],
        "max_salary": [180000, 120000, 210000],
        "normalized_salary": [160000, 105000, 185000],
        "formatted_experience_level": ["MID_SENIOR_LEVEL", "ASSOCIATE", "MID_SENIOR_LEVEL"],
        "skills_desc": ["python django postgresql", "react typescript", "pytorch pandas"],
        "listed_time": [1_700_000_000_000, 1_701_000_000_000, 1_702_000_000_000],
    })
    postings.to_csv(raw / "postings.csv", index=False)

    resumes = pd.DataFrame({
        "ID": [100, 101, 102, 103],
        "Resume_str": ["Senior backend engineer with python and django experience " * 30,
                       "Frontend developer react typescript css " * 20,
                       "Data scientist pytorch ml pandas " * 25,
                       "Junior intern backend python " * 10],
        "Resume_html": ["", "", "", ""],
        "Category": ["INFORMATION-TECHNOLOGY", "DIGITAL-MEDIA", "INFORMATION-TECHNOLOGY", "ENGINEERING"],
    })
    resumes.to_csv(raw / "Resume.csv", index=False)
    return raw


def test_real_adapter_builds_canonical_schema(tmp_path):
    raw = _write_fixtures(tmp_path)
    out = tmp_path / "processed"
    paths = build_real_dataset(raw, out, ontology=SkillOntology(), seed=0)

    jobs = pd.read_csv(paths["jobs"])
    users = pd.read_csv(paths["users"])
    inter = pd.read_csv(paths["interactions"])

    assert set(jobs.columns) >= {"job_id", "title", "category", "seniority", "location",
                                  "skills", "description", "salary_min", "salary_max", "posted_days_ago"}
    assert set(users.columns) >= {"user_id", "primary_category", "seniority", "experience_years",
                                   "preferred_location", "skills", "resume_text"}
    assert set(inter.columns) == {"user_id", "job_id", "action", "timestamp_days_ago"}

    assert len(jobs) == 3
    assert len(users) == 4
    assert len(inter) > 0
    assert set(inter["action"]).issubset({"view", "save", "apply"})
    assert set(inter["user_id"]).issubset(set(users["user_id"]))
    assert set(inter["job_id"]).issubset(set(jobs["job_id"]))
    # HTML stripped from descriptions.
    assert "<p>" not in jobs["description"].iloc[0]
    # Seniority inferred from title keyword.
    assert jobs.loc[jobs["title"].str.contains("Senior"), "seniority"].iloc[0] == "senior"
