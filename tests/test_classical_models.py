"""Tests for content-based, collaborative, popularity, and hybrid recommenders.
Embedder is stubbed with a deterministic hash-based encoder to keep tests offline and fast."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import DataPreprocessor
from src.features.text_features import EmbeddingFeaturizer
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender
from src.models.popularity import PopularityRecommender
from src.models.hybrid import HybridRecommender, hybrid_config_from_settings


# Deterministic offline stand-in for sentence-transformers.
class FakeEmbedder(EmbeddingFeaturizer):
    def __init__(self, dim: int = 32):
        super().__init__(dim=dim)
        self.dim = dim

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
            n = np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
            out = out / n
        return out


@pytest.fixture
def processed(tmp_project):
    return DataPreprocessor(tmp_project).run(persist=False)


@pytest.fixture
def content_model(processed):
    m = ContentBasedRecommender(embedder=FakeEmbedder())
    m.fit(processed.jobs, processed.users, processed.train)
    return m


# Content-based fit produces normalized embeddings and profiles.
def test_content_fit_shapes(content_model, processed):
    art = content_model.artifacts
    assert art.job_embeddings.shape == (len(processed.jobs), 32)
    assert art.user_profiles.shape == (len(processed.users), 32)
    assert np.allclose(np.linalg.norm(art.job_embeddings, axis=1), 1.0, atol=1e-4)


# Content-based recommend returns k valid (job_id, score) pairs.
def test_content_recommend(content_model, processed):
    recs = content_model.recommend(user_id=int(processed.users["user_id"].iloc[0]), k=3)
    assert len(recs) == 3
    job_ids = set(processed.jobs["job_id"])
    assert all(j in job_ids for j, _ in recs)


# Cold-start path produces k recs without requiring a known user.
def test_content_cold_start(content_model, processed):
    recs = content_model.recommend_for_new_user(
        resume="python backend services", skills="python,docker", k=2)
    assert len(recs) == 2


# similar_jobs excludes the query job itself.
def test_content_similar_jobs(content_model, processed):
    jid = int(processed.jobs["job_id"].iloc[0])
    recs = content_model.similar_jobs(jid, k=2)
    assert all(j != jid for j, _ in recs)


# Collaborative SVD fits and predicts in the rating scale.
def test_collaborative_fit_and_predict(processed):
    cf = CollaborativeRecommender(n_factors=8, n_epochs=3).fit(
        processed.train, processed.jobs["job_id"].to_numpy())
    p = cf.predict(int(processed.train["user_id"].iloc[0]), int(processed.train["job_id"].iloc[0]))
    assert 1.0 <= p <= 5.0
    assert cf.knows_user(int(processed.train["user_id"].iloc[0]))


# Popularity ranks jobs with interactions above jobs without.
def test_popularity_ranking(processed):
    pop = PopularityRecommender(recency_halflife_days=30).fit(processed.train, processed.jobs)
    recs = pop.recommend(k=len(processed.jobs))
    assert len(recs) > 0
    scores = [s for _, s in recs]
    assert scores == sorted(scores, reverse=True)


# Hybrid combines three signals and returns k recs for a warm user.
def test_hybrid_recommend(processed, tmp_project):
    content = ContentBasedRecommender(embedder=FakeEmbedder())
    content.fit(processed.jobs, processed.users, processed.train)
    cf = CollaborativeRecommender(n_factors=8, n_epochs=3).fit(
        processed.train, processed.jobs["job_id"].to_numpy())
    pop = PopularityRecommender().fit(processed.train, processed.jobs)
    cfg = hybrid_config_from_settings(tmp_project.models)
    hyb = HybridRecommender(content, cf, pop, cfg).fit(processed.jobs, processed.users, processed.train)

    uid = int(processed.users["user_id"].iloc[0])
    seen = set(processed.train[processed.train["user_id"] == uid]["job_id"])
    n_unseen = len(processed.jobs) - len(seen)
    recs = hyb.recommend(uid, k=n_unseen, exclude_seen=True)
    assert len(recs) == n_unseen
    assert all(j not in seen for j, _ in recs)


# Hybrid.explain returns structured breakdown with matched skills.
def test_hybrid_explain(processed, tmp_project):
    content = ContentBasedRecommender(embedder=FakeEmbedder()).fit(
        processed.jobs, processed.users, processed.train)
    cf = CollaborativeRecommender(n_factors=8, n_epochs=3).fit(
        processed.train, processed.jobs["job_id"].to_numpy())
    pop = PopularityRecommender().fit(processed.train, processed.jobs)
    cfg = hybrid_config_from_settings(tmp_project.models)
    hyb = HybridRecommender(content, cf, pop, cfg).fit(processed.jobs, processed.users, processed.train)

    exp = hyb.explain(user_id=0, job_id=0)
    assert exp["tier"] in {"warm", "lukewarm", "cold"}
    assert set(exp["scores"]) == {"content", "collab", "popularity"}
    assert "python" in exp["matched_skills"]  # user 0 and job 0 both have python


# Cold new-user path through the hybrid works without interactions.
def test_hybrid_new_user(processed, tmp_project):
    content = ContentBasedRecommender(embedder=FakeEmbedder()).fit(
        processed.jobs, processed.users, processed.train)
    cf = CollaborativeRecommender(n_factors=8, n_epochs=3).fit(
        processed.train, processed.jobs["job_id"].to_numpy())
    pop = PopularityRecommender().fit(processed.train, processed.jobs)
    cfg = hybrid_config_from_settings(tmp_project.models)
    hyb = HybridRecommender(content, cf, pop, cfg).fit(processed.jobs, processed.users, processed.train)

    recs = hyb.recommend_for_new_user(resume="react typescript frontend", skills="react,typescript", k=2)
    assert len(recs) == 2
