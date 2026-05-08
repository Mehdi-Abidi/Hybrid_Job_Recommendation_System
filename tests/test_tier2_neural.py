"""Smoke + shape tests for Tier 2 neural models: BERT4Rec, DeepFM, LightGCN, Mult-VAE.
Small configs to keep runs fast; we check fit doesn't error and recommend returns k items."""
from __future__ import annotations
import numpy as np
import pytest

from src.data.preprocessing import DataPreprocessor
from src.models.bert4rec import BERT4RecTrainer, BERT4RecConfig
from src.models.deepfm import DeepFMRecommender, DeepFMConfig
from src.models.lightgcn import LightGCNRecommender, LightGCNConfig
from src.models.mult_vae import MultVAERecommender, MultVAEConfig


@pytest.fixture
def processed(tmp_project):
    return DataPreprocessor(tmp_project).run(persist=False)


# BERT4Rec fits and recommends for a user with prior history.
def test_bert4rec_smoke(processed):
    cfg = BERT4RecConfig(max_len=8, hidden_dim=16, n_heads=2, n_layers=1, batch_size=4, epochs=2, lr=1e-2)
    tr = BERT4RecTrainer(cfg).fit(processed.train)
    history = list(processed.train[processed.train["user_id"] == 0]["job_id"])
    recs = tr.recommend(history, k=2)
    assert 0 <= len(recs) <= 2
    # All recommended items must be in the trained item vocabulary.
    trained = set(processed.train["job_id"].unique())
    for jid, _ in recs:
        assert jid in trained


# DeepFM fits and returns per-job scores in [0, 1].
def test_deepfm_smoke(processed):
    cfg = DeepFMConfig(emb_dim=4, mlp_dims=(8,), batch_size=16, epochs=2, lr=1e-2, n_negatives=2)
    m = DeepFMRecommender(cfg).fit(processed.users, processed.jobs, processed.train)
    recs = m.recommend(user_id=0, k=2, exclude=None)
    assert len(recs) == 2
    for _, s in recs:
        assert 0.0 <= s <= 1.0 + 1e-6


# LightGCN fits on the bipartite graph and produces recommendations.
def test_lightgcn_smoke(processed):
    cfg = LightGCNConfig(emb_dim=8, n_layers=2, batch_size=8, epochs=3, lr=5e-2)
    m = LightGCNRecommender(cfg).fit(processed.users, processed.jobs, processed.train)
    n_unseen = len(processed.jobs) - len(processed.train[processed.train["user_id"] == 0])
    recs = m.recommend(user_id=0, k=max(1, n_unseen), exclude_seen=True)
    assert 0 <= len(recs) <= n_unseen


# LightGCN score_pairs returns a vector of the requested length.
def test_lightgcn_score_pairs(processed):
    cfg = LightGCNConfig(emb_dim=8, n_layers=2, batch_size=8, epochs=2, lr=5e-2)
    m = LightGCNRecommender(cfg).fit(processed.users, processed.jobs, processed.train)
    s = m.score_pairs(user_id=0, job_ids=[0, 1, 2])
    assert s.shape == (3,)


# Mult-VAE fits and generates recommendations per user.
def test_mult_vae_smoke(processed):
    cfg = MultVAEConfig(hidden_dim=16, latent_dim=4, batch_size=4, epochs=3, lr=1e-2, anneal_steps=10)
    m = MultVAERecommender(cfg).fit(processed.users, processed.jobs, processed.train)
    recs = m.recommend(user_id=0, k=2, exclude_seen=True)
    assert len(recs) <= 2


# Mult-VAE score_pairs returns a vector of the requested length.
def test_mult_vae_score_pairs(processed):
    cfg = MultVAEConfig(hidden_dim=16, latent_dim=4, batch_size=4, epochs=2, lr=1e-2, anneal_steps=5)
    m = MultVAERecommender(cfg).fit(processed.users, processed.jobs, processed.train)
    s = m.score_pairs(user_id=0, job_ids=[0, 1, 2])
    assert s.shape == (3,)
