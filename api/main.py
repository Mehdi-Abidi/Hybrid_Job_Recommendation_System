"""FastAPI app exposing the full recommendation system.

Artifacts are loaded lazily on the first request via the AppState singleton. This
keeps uvicorn startup fast and test code can inject its own state."""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import (
    JobSummary, Recommendation, RecommendationResponse, NewUserRequest,
    ExplainResponse, PipelineInspectResponse, QueryRequest, ParsedQueryResponse,
    SalaryRequest, SalaryResponse, SkillGapRequest, SkillGapResponse,
)
from config.settings import load_settings, Settings
from src.data.preprocessing import DataPreprocessor
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender
from src.models.popularity import PopularityRecommender
from src.models.hybrid import HybridRecommender, hybrid_config_from_settings
from src.models.two_tower import TwoTowerTrainer
from src.retrieval.faiss_index import FaissJobIndex
from src.models.ltr_ranker import LTRRanker, LTRConfig, SignalProvider
from src.models.llm_reranker import LLMReranker, LLMConfig
from src.models.salary_predictor import SalaryPredictor
from src.models.skill_gap import SkillGapAnalyzer
from src.ontology.skill_ontology import SkillOntology
from src.nlp.query_understanding import QueryUnderstanding
from src.pipeline.multi_stage import MultiStagePipeline, PipelineStages


class AppState:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.data = None
        self.content: ContentBasedRecommender | None = None
        self.collab: CollaborativeRecommender | None = None
        self.popularity: PopularityRecommender | None = None
        self.hybrid: HybridRecommender | None = None
        self.two_tower: TwoTowerTrainer | None = None
        self.faiss: FaissJobIndex | None = None
        self.ltr: LTRRanker | None = None
        self.llm: LLMReranker | None = None
        self.salary: SalaryPredictor | None = None
        self.ontology: SkillOntology | None = None
        self.query_parser: QueryUnderstanding | None = None
        self.gap: SkillGapAnalyzer | None = None
        self.pipeline: MultiStagePipeline | None = None

    def ensure_loaded(self) -> None:
        if self.data is not None:
            return
        self.data = DataPreprocessor(self.cfg).load()
        root = self.cfg.path("artifacts")
        self.content = ContentBasedRecommender().load(root / "content_based")
        self.collab = CollaborativeRecommender().load(root / "collaborative")
        self.popularity = PopularityRecommender().load(root / "popularity")
        self.hybrid = HybridRecommender(
            self.content, self.collab, self.popularity,
            hybrid_config_from_settings(self.cfg.models),
        ).fit(self.data.jobs, self.data.users, self.data.train)
        self.two_tower = TwoTowerTrainer(self.cfg.models["two_tower"]).load(root / "two_tower")
        self.faiss = FaissJobIndex(
            embedding_dim=self.cfg.models["two_tower"]["embedding_dim"],
            index_type=self.cfg.faiss["index_type"],
            nlist=self.cfg.faiss["nlist"], nprobe=self.cfg.faiss["nprobe"],
        ).load(root / "faiss")
        sp = SignalProvider(self.two_tower, self.content, self.collab, self.popularity)
        self.ltr = LTRRanker(LTRConfig(**{k: self.cfg.models["ltr"][k] for k in
                                          ("objective", "n_estimators", "learning_rate", "max_depth")}), sp) \
            .load(root / "ltr", self.data.users, self.data.jobs)
        self.llm = LLMReranker(LLMConfig(
            model=self.cfg.models["llm"]["model"],
            rerank_top_k=self.cfg.models["llm"]["rerank_top_k"],
            output_top_n=self.cfg.models["llm"]["output_top_n"],
            max_tokens=self.cfg.models["llm"]["max_tokens"],
        ))
        self.ontology = SkillOntology.load(root / "ontology") if (root / "ontology" / "ontology.json").exists() \
            else SkillOntology()
        try:
            self.salary = SalaryPredictor().load(root / "salary")
        except FileNotFoundError:
            self.salary = None
        self.query_parser = QueryUnderstanding(ontology=self.ontology)
        self.gap = SkillGapAnalyzer(self.ontology)
        history = self.data.train.groupby("user_id")["job_id"].apply(lambda s: {int(x) for x in s}).to_dict()
        self.pipeline = MultiStagePipeline(
            PipelineStages(two_tower=self.two_tower, faiss=self.faiss, content=self.content,
                           collab=self.collab, popularity=self.popularity, ltr=self.ltr, llm=self.llm),
            self.data.users, self.data.jobs, user_history=history,
            n_retrieve=self.cfg.models["llm"]["rerank_top_k"] * 5,
            n_rank=self.cfg.models["llm"]["rerank_top_k"],
            n_final=self.cfg.models["llm"]["output_top_n"],
        )


_STATE: AppState | None = None


def get_state() -> AppState:
    global _STATE
    if _STATE is None:
        _STATE = AppState(load_settings())
    _STATE.ensure_loaded()
    return _STATE


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # lazy load — avoid forcing heavy init at startup


app = FastAPI(title="Hybrid Job Recommender", version="1.0", lifespan=lifespan)


def _job_summary(row: pd.Series) -> JobSummary:
    return JobSummary(
        job_id=int(row["job_id"]), title=str(row.get("title", "")),
        category=str(row.get("category", "")) or None,
        seniority=str(row.get("seniority", "")) or None,
        location=str(row.get("location", "")) or None,
        skills=str(row.get("skills", "")) or None,
        salary_min=float(row.get("salary_min") or 0) or None,
        salary_max=float(row.get("salary_max") or 0) or None,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def recommend(user_id: int, k: int = 10, model: str = "hybrid"):
    s = get_state()
    if model == "hybrid":
        recs = s.hybrid.recommend(user_id, k=k)
    elif model == "content":
        if user_id not in s.content._user_index:
            raise HTTPException(404, "user not found in content model")
        recs = s.content.recommend(user_id, k=k)
    elif model == "collab":
        recs = s.collab.recommend(user_id, k=k)
    elif model == "popularity":
        recs = s.popularity.recommend(k=k)
    else:
        raise HTTPException(400, f"unknown model: {model}")
    return RecommendationResponse(
        user_id=user_id, model=model,
        recommendations=[Recommendation(job_id=j, score=sc) for j, sc in recs],
    )


@app.post("/recommend/new-user", response_model=RecommendationResponse)
def recommend_new_user(req: NewUserRequest):
    s = get_state()
    recs = s.hybrid.recommend_for_new_user(resume=req.resume, skills=req.skills, k=req.k)
    return RecommendationResponse(
        recommendations=[Recommendation(job_id=j, score=sc) for j, sc in recs],
        model="hybrid_cold_start",
    )


@app.get("/similar-jobs/{job_id}", response_model=list[JobSummary])
def similar_jobs(job_id: int, k: int = 10):
    s = get_state()
    ids = [j for j, _ in s.content.similar_jobs(job_id, k=k)]
    rows = s.data.jobs.set_index("job_id").loc[ids].reset_index()
    return [_job_summary(r) for _, r in rows.iterrows()]


@app.get("/explain/{user_id}/{job_id}", response_model=ExplainResponse)
def explain(user_id: int, job_id: int):
    s = get_state()
    out = s.hybrid.explain(user_id, job_id)
    return ExplainResponse(**out)


@app.get("/recommend/multi-stage/{user_id}", response_model=RecommendationResponse)
def multi_stage(user_id: int, exclude_seen: bool = True):
    s = get_state()
    recs = s.pipeline.recommend(user_id, exclude_seen=exclude_seen)
    return RecommendationResponse(
        user_id=user_id, model="multi_stage",
        recommendations=[Recommendation(job_id=r.job_id, score=r.score,
                                        explanation=r.explanation, stage_scores=r.stage_scores)
                         for r in recs],
    )


@app.get("/pipeline/inspect/{user_id}", response_model=PipelineInspectResponse)
def inspect(user_id: int):
    s = get_state()
    return PipelineInspectResponse(**s.pipeline.inspect(user_id))


@app.post("/query/parse", response_model=ParsedQueryResponse)
def parse_query(req: QueryRequest):
    s = get_state()
    p = s.query_parser.parse(req.query)
    return ParsedQueryResponse(skills=p.skills, seniority=p.seniority, category=p.category,
                               location=p.location, min_salary=p.min_salary, remote=p.remote)


@app.post("/salary/predict", response_model=SalaryResponse)
def predict_salary(req: SalaryRequest):
    s = get_state()
    if s.salary is None:
        raise HTTPException(503, "salary model not available")
    out = s.salary.predict(category=req.category, seniority=req.seniority,
                           location=req.location, skills=req.skills)
    return SalaryResponse(salary_min=out.salary_min, salary_max=out.salary_max, midpoint=out.midpoint)


@app.post("/skill-gap", response_model=SkillGapResponse)
def skill_gap(req: SkillGapRequest):
    s = get_state()
    from src.features.structured_features import parse_skills
    r = s.gap.analyze(parse_skills(req.user_skills), parse_skills(req.target_skills))
    return SkillGapResponse(matched=r.matched, missing=r.missing, adjacent=r.adjacent,
                            match_rate=r.match_rate, adjacent_rate=r.adjacent_rate)
