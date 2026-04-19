"""Tests for Tier 3 domain modules: resume NER, skill ontology, salary predictor,
career path, skill gap, query understanding (regex fallback)."""
from __future__ import annotations
from unittest.mock import MagicMock
import pandas as pd
import pytest

from src.nlp.resume_ner import ResumeNER
from src.ontology.skill_ontology import SkillOntology
from src.models.salary_predictor import SalaryPredictor
from src.models.career_path import CareerPathPredictor
from src.models.skill_gap import SkillGapAnalyzer
from src.nlp.query_understanding import QueryUnderstanding


# Resume NER extracts experience years and known skills.
def test_resume_ner_basic():
    text = "Senior backend engineer with 7 years of experience. Skills: Python, Docker, Kubernetes. MS in CS."
    ner = ResumeNER(skill_vocab=["python", "docker", "kubernetes", "rust"])
    out = ner.parse(text)
    assert out.experience_years == 7
    assert set(out.skills) == {"python", "docker", "kubernetes"}
    assert any("ms" in e.lower() for e in out.education)


# NER without vocab still extracts years.
def test_resume_ner_no_vocab():
    out = ResumeNER().parse("3+ years as a data scientist")
    assert out.experience_years == 3
    assert "data scientist" in out.titles


# Ontology normalization handles aliases.
def test_ontology_normalize():
    o = SkillOntology()
    assert o.normalize("JS") == "javascript"
    assert o.normalize("postgres") == "postgresql"
    assert o.normalize("python") == "python"


# Skills in the same category have high relatedness.
def test_ontology_relatedness_shared_category():
    o = SkillOntology()
    r = o.relatedness("python", "java")
    assert r >= 0.5


# Unrelated skills have lower relatedness.
def test_ontology_relatedness_distant():
    o = SkillOntology()
    a = o.relatedness("python", "python")
    b = o.relatedness("react", "kubernetes")
    assert a > b


# Category inference returns the dominant category.
def test_ontology_infer_category():
    o = SkillOntology()
    cat, conf = o.infer_category(["python", "django", "postgresql"])
    assert cat == "backend"
    assert conf > 0


# Salary predictor trains and returns a valid range.
def test_salary_predictor(tmp_project):
    from src.data.preprocessing import DataPreprocessor
    data = DataPreprocessor(tmp_project).run(persist=False)
    pred = SalaryPredictor(n_estimators=10, max_depth=3).fit(data.jobs)
    out = pred.predict(category="backend", seniority="mid", location="Remote", skills="python,docker")
    assert out.salary_min > 0
    assert out.salary_max >= out.salary_min
    assert out.midpoint == pytest.approx((out.salary_min + out.salary_max) / 2)


# Career path: a user with two applies builds a title transition.
def test_career_path():
    jobs = pd.DataFrame([
        {"job_id": 0, "title": "Backend Engineer"},
        {"job_id": 1, "title": "Senior Software Engineer"},
    ])
    interactions = pd.DataFrame([
        {"user_id": 0, "job_id": 0, "action": "apply", "rating": 5, "timestamp_days_ago": 100},
        {"user_id": 0, "job_id": 1, "action": "apply", "rating": 5, "timestamp_days_ago": 10},
    ])
    cp = CareerPathPredictor().fit(interactions, jobs)
    steps = cp.predict_next("backend engineer", k=3)
    assert len(steps) >= 1
    assert steps[0].to_title == "senior software engineer"


# Skill gap: matched / missing / adjacent partition target skills correctly.
def test_skill_gap():
    o = SkillOntology()
    gap = SkillGapAnalyzer(o, relatedness_threshold=0.5)
    r = gap.analyze(user_skills=["python", "docker"], target_skills=["python", "kubernetes", "rust"])
    assert "python" in r.matched
    assert "kubernetes" in r.missing or "rust" in r.missing
    assert r.match_rate == pytest.approx(1 / 3)


# Query understanding regex fallback extracts seniority, location, min salary.
def test_query_understanding_regex_fallback():
    q = QueryUnderstanding(ontology=SkillOntology(), api_key=None, client=None)
    parsed = q.parse("senior python backend role in NYC that pays at least 180k, remote")
    assert parsed.seniority == "senior"
    assert "python" in parsed.skills
    assert parsed.min_salary == 180000
    assert parsed.remote is True
    assert parsed.location and "nyc" in parsed.location.lower()


# Query understanding with mocked Claude client parses JSON correctly.
def test_query_understanding_with_mocked_client():
    mock_client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = (
        '{"skills": ["python", "react"], "seniority": "senior", "category": "frontend", '
        '"location": "Remote", "min_salary": 150000, "remote": true}'
    )
    mock_client.chat.completions.create.return_value = resp
    q = QueryUnderstanding(ontology=SkillOntology(), client=mock_client)
    parsed = q.parse("anything")
    assert parsed.seniority == "senior"
    assert set(parsed.skills) == {"python", "react"}
    assert parsed.min_salary == 150000
