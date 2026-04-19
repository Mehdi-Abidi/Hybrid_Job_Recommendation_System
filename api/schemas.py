"""Pydantic request/response schemas for the recommendation API."""
from __future__ import annotations
from pydantic import BaseModel, Field


class JobSummary(BaseModel):
    job_id: int
    title: str
    category: str | None = None
    seniority: str | None = None
    location: str | None = None
    skills: str | None = None
    salary_min: float | None = None
    salary_max: float | None = None


class Recommendation(BaseModel):
    job_id: int
    score: float
    explanation: str | None = None
    stage_scores: dict[str, float] = Field(default_factory=dict)


class RecommendationResponse(BaseModel):
    user_id: int | None = None
    recommendations: list[Recommendation]
    model: str = "hybrid"


class NewUserRequest(BaseModel):
    resume: str = ""
    skills: str = ""
    k: int = 10


class ExplainResponse(BaseModel):
    tier: str
    weights: dict[str, float]
    scores: dict[str, float]
    matched_skills: list[str]
    job_title: str


class PipelineInspectResponse(BaseModel):
    retrieval: list[tuple[int, float]]
    ranking: list[tuple[int, float]]
    final: list[tuple[int, float, str]]


class QueryRequest(BaseModel):
    query: str
    k: int = 10


class ParsedQueryResponse(BaseModel):
    skills: list[str]
    seniority: str | None = None
    category: str | None = None
    location: str | None = None
    min_salary: int | None = None
    remote: bool | None = None


class SalaryRequest(BaseModel):
    category: str = ""
    seniority: str = ""
    location: str = ""
    skills: str = ""


class SalaryResponse(BaseModel):
    salary_min: float
    salary_max: float
    midpoint: float


class SkillGapRequest(BaseModel):
    user_skills: str
    target_skills: str


class SkillGapResponse(BaseModel):
    matched: list[str]
    missing: list[str]
    adjacent: list[tuple[str, str, float]]
    match_rate: float
    adjacent_rate: float
