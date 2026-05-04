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
from src.models.reciprocal import BilateralScorer
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
    bilateral: BilateralScorer | None = None


@dataclass
class Recommendation:
    job_id: int
    score: float
    explanation: str = ""
    stage_scores: dict[str, float] = field(default_factory=dict)


class MultiStagePipeline:
    def __init__(self, stages: PipelineStages, users: pd.DataFrame, jobs: pd.DataFrame,
                 user_history: dict[int, set[int]] | None = None,
                 n_retrieve: int = 500, n_rank: int = 20, n_final: int = 10,
                 mmr_lambda: float = 0.7, max_posted_days: int = 90):
        self.s = stages
        self.users = users
        self.jobs = jobs
        self.user_history = user_history or {}
        self.n_retrieve = n_retrieve
        self.n_rank = n_rank
        self.n_final = n_final
        self.mmr_lambda = mmr_lambda
        # Active-job filter: drop postings older than max_posted_days. None = disabled.
        if "posted_days_ago" in jobs.columns and max_posted_days is not None:
            self._active_jobs = set(int(j) for j in jobs.loc[
                jobs["posted_days_ago"] <= max_posted_days, "job_id"
            ])
        else:
            self._active_jobs = None
        # Pre-compute job embedding lookup for MMR diversity (uses two-tower if wired).
        if stages.two_tower is not None:
            art = stages.two_tower.artifacts
            self._j_row = {int(j): i for i, j in enumerate(art.job_ids)}
            self._j_emb = stages.two_tower.job_embeddings()
        else:
            self._j_row = {}
            self._j_emb = None

    # Stage 1: retrieve candidates via two-tower+FAISS, optionally merged with classical sources.
    def _retrieve(self, user_id: int) -> list[tuple[int, float]]:
        cands: dict[int, float] = {}
        if self.s.two_tower is not None and self.s.faiss is not None:
            art = self.s.two_tower.artifacts
            u_id_to_row = {int(u): i for i, u in enumerate(art.user_ids)}
            u_idx = u_id_to_row.get(int(user_id), -1)
            if u_idx >= 0:
                ue = self.s.two_tower.user_embeddings()[u_idx]
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

    # Compute bilateral scores for a list of job_ids (or None if no scorer wired).
    # Returned dicts are keyed by job_id; absent keys mean the scorer wasn't available.
    def _bilateral_map(self, user_id: int, job_ids: list[int]) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
        if self.s.bilateral is None or not job_ids:
            return {}, {}, {}
        s_uj, s_ju, bilat = self.s.bilateral.score(int(user_id), [int(j) for j in job_ids])
        return (
            {int(j): float(s_uj[i]) for i, j in enumerate(job_ids)},
            {int(j): float(s_ju[i]) for i, j in enumerate(job_ids)},
            {int(j): float(bilat[i]) for i, j in enumerate(job_ids)},
        )

    # Stage 3: LLM re-rank + explanations. Bilateral scores (when available) are
    # surfaced in stage_scores and embedded in the LLM prompt for richer reasoning.
    def _rerank(self, user_id: int, ranked: list[tuple[int, float]]) -> list[Recommendation]:
        ranked_ids = [j for j, _ in ranked]
        s_uj_map, s_ju_map, bilat_map = self._bilateral_map(user_id, ranked_ids)

        if self.s.llm is None:
            recs: list[Recommendation] = []
            for j, s in ranked[:self.n_final]:
                ss: dict[str, float] = {"ltr": float(s)}
                if j in bilat_map:
                    ss["s_user_to_job"] = s_uj_map[j]
                    ss["s_job_to_user"] = s_ju_map[j]
                    ss["bilateral"] = bilat_map[j]
                recs.append(Recommendation(job_id=j, score=float(s), explanation="", stage_scores=ss))
            return recs

        user_row = self.users[self.users["user_id"] == user_id]
        user_profile = user_row.iloc[0].to_dict() if not user_row.empty else {"user_id": user_id}
        jobs_by_id = self.jobs.set_index("job_id")
        candidates = []
        for jid, prior_score in ranked:
            if jid not in jobs_by_id.index:
                continue
            jr = jobs_by_id.loc[jid]
            cand: dict[str, Any] = {
                "job_id": int(jid), "title": str(jr.get("title", "")),
                "category": str(jr.get("category", "")), "seniority": str(jr.get("seniority", "")),
                "location": str(jr.get("location", "")), "skills": str(jr.get("skills", "")),
                "prior_score": float(prior_score),
            }
            if jid in bilat_map:
                cand["bilateral_score"] = bilat_map[jid]
            candidates.append(cand)
        reranked = self.s.llm.rerank(user_profile, candidates, n=self.n_final)
        prior_map = dict(ranked)
        recs = []
        for jid, score, reason in reranked:
            ss = {"ltr": float(prior_map.get(jid, 0.0)), "llm": float(score)}
            if jid in bilat_map:
                ss["s_user_to_job"] = s_uj_map[jid]
                ss["s_job_to_user"] = s_ju_map[jid]
                ss["bilateral"] = bilat_map[jid]
            recs.append(Recommendation(job_id=jid, score=float(score), explanation=reason, stage_scores=ss))
        return recs

    # MMR diversity rerank: trade off relevance (LTR score) against similarity to
    # already-selected items, computed in two-tower job-embedding space. Standard
    # post-LTR step in production rec systems (YouTube CIKM 2018 used DPP — MMR
    # is the simpler well-understood predecessor with proven engagement lift).
    def _diversify_mmr(self, ranked: list[tuple[int, float]]) -> list[tuple[int, float]]:
        if self._j_emb is None or len(ranked) <= self.n_final:
            return ranked[: self.n_final]
        selected: list[tuple[int, float]] = []
        pool = list(ranked)
        sel_emb_rows: list[int] = []
        while pool and len(selected) < self.n_final:
            if not selected:
                # Pick the highest-scoring candidate to seed.
                best_idx = max(range(len(pool)), key=lambda i: pool[i][1])
            else:
                sel_emb = self._j_emb[sel_emb_rows]
                best_idx, best_mmr = 0, -np.inf
                for i, (jid, s) in enumerate(pool):
                    row = self._j_row.get(int(jid))
                    if row is None:
                        mmr_score = s  # unknown emb — score stands alone
                    else:
                        sim = float(np.max(sel_emb @ self._j_emb[row]))
                        mmr_score = self.mmr_lambda * s - (1 - self.mmr_lambda) * sim
                    if mmr_score > best_mmr:
                        best_mmr, best_idx = mmr_score, i
            picked = pool.pop(best_idx)
            selected.append(picked)
            row = self._j_row.get(int(picked[0]))
            if row is not None:
                sel_emb_rows.append(row)
        return selected

    def recommend(self, user_id: int, exclude_seen: bool = True) -> list[Recommendation]:
        retrieved = self._retrieve(int(user_id))
        # Business rules: drop applied/seen jobs and expired postings BEFORE ranking.
        if exclude_seen:
            seen = self.user_history.get(int(user_id), set())
            retrieved = [(j, s) for j, s in retrieved if j not in seen]
        if self._active_jobs is not None:
            retrieved = [(j, s) for j, s in retrieved if int(j) in self._active_jobs]
        ranked = self._rank(int(user_id), retrieved)
        diversified = self._diversify_mmr(ranked)
        recs = self._rerank(int(user_id), diversified)
        # Ensure retrieval and LTR scores are visible even if LLM added explanation.
        retrieve_map = dict(retrieved)
        for r in recs:
            r.stage_scores.setdefault("retrieve", float(retrieve_map.get(r.job_id, 0.0)))
        return recs

    # Introspection: return each stage's candidates for debugging / UI inspection.
    # Bilateral scores are included when a BilateralScorer is wired in — empty otherwise.
    def inspect(self, user_id: int) -> dict[str, Any]:
        retrieved = self._retrieve(int(user_id))
        ranked = self._rank(int(user_id), retrieved)
        final = self._rerank(int(user_id), ranked)
        out: dict[str, Any] = {
            "retrieval": retrieved,
            "ranking": ranked,
            "final": [(r.job_id, r.score, r.explanation) for r in final],
        }
        if self.s.bilateral is not None and ranked:
            _, _, bilat_map = self._bilateral_map(user_id, [j for j, _ in ranked])
            out["bilateral"] = sorted(bilat_map.items(), key=lambda x: -x[1])
        return out
