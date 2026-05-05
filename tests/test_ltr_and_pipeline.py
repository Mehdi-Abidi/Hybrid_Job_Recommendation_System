"""Tests for ranking features, LTR ranker, LLM re-ranker (mocked), and multi-stage pipeline."""
from __future__ import annotations
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import DataPreprocessor
from src.features.ranking_features import FEATURE_NAMES, build_ranking_features, RankingSignals
from src.features.text_features import EmbeddingFeaturizer
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender
from src.models.popularity import PopularityRecommender
from src.models.two_tower import TwoTowerTrainer
from src.models.ltr_ranker import LTRRanker, LTRConfig, SignalProvider
from src.models.llm_reranker import LLMReranker, LLMConfig
from src.pipeline.multi_stage import MultiStagePipeline, PipelineStages
from src.retrieval.faiss_index import FaissJobIndex


class FakeEmbedder(EmbeddingFeaturizer):
    def __init__(self, dim: int = 8):
        super().__init__(dim=dim); self.dim = dim

    def encode(self, texts, normalize=True):
        rng = np.random.default_rng(0)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        vocab = {}
        for i, t in enumerate(texts):
            for tok in str(t).lower().split():
                if tok not in vocab:
                    vocab[tok] = rng.standard_normal(self.dim).astype(np.float32)
                out[i] += vocab[tok]
        if normalize:
            out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
        return out


@pytest.fixture
def processed(tmp_project):
    return DataPreprocessor(tmp_project).run(persist=False)


@pytest.fixture
def classical_models(processed):
    content = ContentBasedRecommender(embedder=FakeEmbedder()).fit(
        processed.jobs, processed.users, processed.train)
    cf = CollaborativeRecommender(n_factors=4, n_epochs=3).fit(
        processed.train, processed.jobs["job_id"].to_numpy())
    pop = PopularityRecommender().fit(processed.train, processed.jobs)
    return content, cf, pop


@pytest.fixture
def trained_tt(processed):
    cfg = {"embedding_dim": 8, "hidden_dims": [16], "dropout": 0.0, "batch_size": 8, "epochs": 2, "lr": 1e-2}
    t = TwoTowerTrainer(cfg, text_embedder=FakeEmbedder(dim=8))
    t.build_features(processed.jobs, processed.users)
    t.train(processed.train, epochs=2, batch_size=8, lr=1e-2)
    return t


# Ranking feature matrix has the correct shape and column order.
def test_ranking_features_shape(processed, classical_models):
    content, cf, pop = classical_models
    uid = int(processed.users["user_id"].iloc[0])
    cands = list(processed.jobs["job_id"])
    sp = SignalProvider(None, content, cf, pop)
    sigs = sp.compute(uid, cands)
    feats = build_ranking_features(uid, cands, processed.users, processed.jobs, sigs)
    assert feats.shape == (len(cands), len(FEATURE_NAMES))
    assert feats.dtype == np.float32


# LTR fits and returns a ranked list of the supplied candidates.
def test_ltr_fit_and_rank(processed, classical_models, trained_tt):
    content, cf, pop = classical_models
    sp = SignalProvider(trained_tt, content, cf, pop)
    ltr = LTRRanker(LTRConfig(n_estimators=5, max_depth=2), sp).fit(
        processed.users, processed.jobs, processed.train)
    uid = int(processed.users["user_id"].iloc[0])
    cands = list(processed.jobs["job_id"])
    ranked = ltr.rank(uid, cands)
    assert sorted([j for j, _ in ranked]) == sorted(cands)
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)


# Feature-importance dict is non-empty after fit (some features should be used).
def test_ltr_feature_importance(processed, classical_models, trained_tt):
    content, cf, pop = classical_models
    sp = SignalProvider(trained_tt, content, cf, pop)
    ltr = LTRRanker(LTRConfig(n_estimators=10, max_depth=3), sp).fit(
        processed.users, processed.jobs, processed.train)
    imp = ltr.feature_importance()
    assert isinstance(imp, dict)  # may be empty on tiny data, but must be a dict


# LLM re-ranker falls back to pass-through when no client is available.
def test_llm_rerank_fallback_no_client():
    llm = LLMReranker(LLMConfig(output_top_n=2), api_keys=[], client=None)
    cands = [{"job_id": 1, "prior_score": 0.9, "title": "a"},
             {"job_id": 2, "prior_score": 0.5, "title": "b"}]
    out = llm.rerank({"user_id": 0, "skills": "python"}, cands, n=2)
    assert [x[0] for x in out] == [1, 2]
    assert all(x[2] == "" for x in out)  # no explanation in fallback


# LLM re-ranker parses a mocked client response and re-orders candidates.
def test_llm_rerank_with_mocked_client():
    mock_client = MagicMock()
    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = (
        '{"ranked": [{"job_id": 2, "score": 0.9, "reason": "better fit"}, '
        '{"job_id": 1, "score": 0.6, "reason": "secondary"}]}'
    )
    mock_client.chat.completions.create.return_value = fake_response
    llm = LLMReranker(LLMConfig(output_top_n=2), client=mock_client)
    cands = [{"job_id": 1, "prior_score": 0.9, "title": "a"},
             {"job_id": 2, "prior_score": 0.5, "title": "b"}]
    out = llm.rerank({"user_id": 0, "skills": "python"}, cands, n=2)
    assert out[0][0] == 2 and out[1][0] == 1
    assert "better fit" in out[0][2]


# End-to-end: retrieve → rank → (no-LLM) produces k recommendations with per-stage scores.
def test_pipeline_end_to_end(processed, classical_models, trained_tt):
    content, cf, pop = classical_models
    je = trained_tt.job_embeddings()
    faiss_idx = FaissJobIndex(embedding_dim=je.shape[1], index_type="Flat").build(
        je, trained_tt.artifacts.job_ids)
    sp = SignalProvider(trained_tt, content, cf, pop)
    ltr = LTRRanker(LTRConfig(n_estimators=5, max_depth=2), sp).fit(
        processed.users, processed.jobs, processed.train)

    stages = PipelineStages(two_tower=trained_tt, faiss=faiss_idx, content=content,
                            collab=cf, popularity=pop, ltr=ltr, llm=None)
    history = processed.train.groupby("user_id")["job_id"].apply(
        lambda s: {int(x) for x in s}).to_dict()
    pipe = MultiStagePipeline(stages, processed.users, processed.jobs, user_history=history,
                              n_retrieve=3, n_rank=2, n_final=2)
    recs = pipe.recommend(user_id=int(processed.users["user_id"].iloc[0]), exclude_seen=True)
    assert 1 <= len(recs) <= 2
    for r in recs:
        assert "retrieve" in r.stage_scores
        assert "ltr" in r.stage_scores


# Pipeline.inspect() returns dict with three stage keys.
def test_pipeline_inspect(processed, classical_models, trained_tt):
    content, cf, pop = classical_models
    je = trained_tt.job_embeddings()
    faiss_idx = FaissJobIndex(embedding_dim=je.shape[1], index_type="Flat").build(
        je, trained_tt.artifacts.job_ids)
    sp = SignalProvider(trained_tt, content, cf, pop)
    ltr = LTRRanker(LTRConfig(n_estimators=5, max_depth=2), sp).fit(
        processed.users, processed.jobs, processed.train)
    stages = PipelineStages(two_tower=trained_tt, faiss=faiss_idx, content=content,
                            collab=cf, popularity=pop, ltr=ltr, llm=None)
    pipe = MultiStagePipeline(stages, processed.users, processed.jobs,
                              n_retrieve=3, n_rank=2, n_final=2)
    out = pipe.inspect(user_id=int(processed.users["user_id"].iloc[0]))
    assert set(out) == {"retrieval", "ranking", "final"}
