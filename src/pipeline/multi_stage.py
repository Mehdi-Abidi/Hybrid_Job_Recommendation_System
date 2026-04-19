"""Multi-stage recommendation pipeline: Retrieval → Ranking → LLM re-rank.

Stage 1 (Retrieval): Two-Tower + FAISS → top-N1 candidates. Falls back / merges
with content+collab+popularity candidates for robustness.
Stage 2 (Ranking): LambdaMART LTR rescores top-N1 → top-N2.
Stage 3 (Re-rank): LLM re-orders top-N2 → top-K with explanations.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd

from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender
from src.models.popularity import PopularityRecommender
from src.models.two_tower import TwoTowerTrainer
from src.models.ltr_ranker import LTRRanker
from src.models.llm_reranker import LLMReranker
from src.retrieval.faiss_index import FaissJobIndex
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class PipelineStages:
    two_tower: TwoTowerTrainer | None = None
    faiss: FaissJobIndex | None = None
    content: ContentBasedRecommender | None = None
    collab: CollaborativeRecommender | None = None
    popularity: PopularityRecommender | None = None
    ltr: LTRRanker | None = None
    llm: LLMReranker | None = None


@dataclass
class Recommendation:
    job_id: int
    score: float
    explanation: str = ""
    stage_scores: dict[str, float] = field(default_factory=dict)


class MultiStagePipeline:
    def __init__(self, stages: PipelineStages, users: pd.DataFrame, jobs: pd.DataFrame,
                 user_history: dict[int, set[int]] | None = None,
                 n_retrieve: int = 500, n_rank: int = 20, n_final: int = 10):
        self.s = stages
        self.users = users
        self.jobs = jobs
        self.user_history = user_history or {}
        self.n_retrieve = n_retrieve
        self.n_rank = n_rank
        self.n_final = n_final

    # Stage 1: retrieve candidates via two-tower+FAISS, optionally merged with classical sources.
    def _retrieve(self, user_id: int) -> list[tuple[int, float]]:
        cands: dict[int, float] = {}
        if self.s.two_tower is not None and self.s.faiss is not None:
            art = self.s.two_tower.artifacts
            u_pos = np.where(art.user_ids == user_id)[0]
            if len(u_pos) > 0:
                ue = self.s.two_tower.user_embeddings()[u_pos[0]]
                for jid, score in self.s.faiss.search(ue, k=self.n_retrieve):
                    cands[jid] = max(cands.get(jid, -np.inf), score)
        # Fallback/merge: classical signals ensure coverage.
        if len(cands) < self.n_retrieve:
            fill = self.n_retrieve - len(cands)
            if self.s.content is not None and user_id in self.s.content._user_index:
                for jid, s in self.s.content.recommend(user_id, k=fill):
                    cands.setdefault(jid, s)
            if self.s.popularity is not None:
                for jid, s in self.s.popularity.recommend(k=fill):
                    cands.setdefault(jid, s)
        return sorted(cands.items(), key=lambda x: -x[1])[:self.n_retrieve]

    # Stage 2: LTR rescore.
    def _rank(self, user_id: int, retrieved: list[tuple[int, float]]) -> list[tuple[int, float]]:
        cand_ids = [jid for jid, _ in retrieved]
        if self.s.ltr is None or self.s.ltr.booster is None:
            return retrieved[:self.n_rank]
        return self.s.ltr.rank(user_id, cand_ids)[:self.n_rank]

    # Stage 3: LLM re-rank + explanations.
    def _rerank(self, user_id: int, ranked: list[tuple[int, float]]) -> list[Recommendation]:
        if self.s.llm is None:
            return [Recommendation(job_id=j, score=s, explanation="", stage_scores={"ltr": s})
                    for j, s in ranked[:self.n_final]]

        user_row = self.users[self.users["user_id"] == user_id]
        user_profile = user_row.iloc[0].to_dict() if not user_row.empty else {"user_id": user_id}
        jobs_by_id = self.jobs.set_index("job_id")
        candidates = []
        for jid, prior_score in ranked:
            if jid not in jobs_by_id.index:
                continue
            jr = jobs_by_id.loc[jid]
            candidates.append({
                "job_id": int(jid), "title": str(jr.get("title", "")),
                "category": str(jr.get("category", "")), "seniority": str(jr.get("seniority", "")),
                "location": str(jr.get("location", "")), "skills": str(jr.get("skills", "")),
                "prior_score": float(prior_score),
            })
        reranked = self.s.llm.rerank(user_profile, candidates, n=self.n_final)
        prior_map = dict(ranked)
        return [
            Recommendation(
                job_id=jid, score=float(score), explanation=reason,
                stage_scores={"ltr": float(prior_map.get(jid, 0.0)), "llm": float(score)},
            )
            for jid, score, reason in reranked
        ]

    def recommend(self, user_id: int, exclude_seen: bool = True) -> list[Recommendation]:
        retrieved = self._retrieve(int(user_id))
        if exclude_seen:
            seen = self.user_history.get(int(user_id), set())
            retrieved = [(j, s) for j, s in retrieved if j not in seen]
        ranked = self._rank(int(user_id), retrieved)
        recs = self._rerank(int(user_id), ranked)
        # Ensure retrieval and LTR scores are visible even if LLM added explanation.
        retrieve_map = dict(retrieved)
        for r in recs:
            r.stage_scores.setdefault("retrieve", float(retrieve_map.get(r.job_id, 0.0)))
        return recs

    # Introspection: return each stage's candidates for debugging / UI inspection.
    def inspect(self, user_id: int) -> dict[str, Any]:
        retrieved = self._retrieve(int(user_id))
        ranked = self._rank(int(user_id), retrieved)
        final = self._rerank(int(user_id), ranked)
        return {
            "retrieval": retrieved,
            "ranking": ranked,
            "final": [(r.job_id, r.score, r.explanation) for r in final],
        }
