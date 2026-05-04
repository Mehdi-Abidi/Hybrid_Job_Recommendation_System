"""MMR diversity rerank correctness."""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.pipeline.multi_stage import MultiStagePipeline, PipelineStages


class _StubTwoTower:
    """Minimal stand-in: provides job_embeddings + artifacts.job_ids."""
    def __init__(self, job_emb: np.ndarray, job_ids: np.ndarray):
        self._job_emb = job_emb
        from types import SimpleNamespace
        self.artifacts = SimpleNamespace(job_ids=job_ids, user_ids=np.arange(0))

    def job_embeddings(self): return self._job_emb
    def user_embeddings(self): return np.zeros((0, self._job_emb.shape[1]))


def _pipe(job_emb, job_ids, lambda_=0.5):
    jobs = pd.DataFrame({"job_id": job_ids})
    users = pd.DataFrame({"user_id": [0]})
    stages = PipelineStages(two_tower=_StubTwoTower(job_emb, job_ids))
    return MultiStagePipeline(stages, users=users, jobs=jobs,
                              n_final=3, mmr_lambda=lambda_, max_posted_days=None)


def test_mmr_picks_diverse_when_top_are_duplicates():
    """Top 3 by score are near-duplicates; MMR with low λ should mix in a different one."""
    job_emb = np.array([
        [1.0, 0.0],   # 0
        [0.99, 0.0],  # 1 — near-duplicate of 0
        [0.98, 0.0],  # 2 — near-duplicate of 0
        [0.0, 1.0],   # 3 — orthogonal
    ], dtype=np.float32)
    pipe = _pipe(job_emb, np.array([0, 1, 2, 3]), lambda_=0.3)
    ranked = [(0, 1.0), (1, 0.95), (2, 0.94), (3, 0.5)]
    out = pipe._diversify_mmr(ranked)
    out_ids = [j for j, _ in out]
    # First pick is always the top score
    assert out_ids[0] == 0
    # With low λ (favouring diversity), the orthogonal item 3 should beat near-dup 1/2
    assert 3 in out_ids


def test_mmr_picks_high_relevance_when_lambda_high():
    """High λ → MMR converges to plain top-k by score."""
    job_emb = np.array([
        [1.0, 0.0], [0.99, 0.0], [0.98, 0.0], [0.0, 1.0],
    ], dtype=np.float32)
    pipe = _pipe(job_emb, np.array([0, 1, 2, 3]), lambda_=0.99)
    ranked = [(0, 1.0), (1, 0.95), (2, 0.94), (3, 0.5)]
    out = pipe._diversify_mmr(ranked)
    assert [j for j, _ in out] == [0, 1, 2]


def test_mmr_returns_n_final_or_fewer():
    job_emb = np.eye(5, dtype=np.float32)
    pipe = _pipe(job_emb, np.arange(5))
    ranked = [(i, 1.0 - 0.1 * i) for i in range(5)]
    out = pipe._diversify_mmr(ranked)
    assert len(out) == 3  # n_final=3
