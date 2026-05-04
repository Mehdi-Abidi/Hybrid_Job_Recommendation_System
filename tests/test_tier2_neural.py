"""Smoke test for the surviving tier-2 model: BERT4Rec.

DeepFM/LightGCN/Mult-VAE were trimmed in Task 9 — they were trained but never
consumed downstream. LiRank (KDD 2024) and similar production systems ship
exactly one transformer-based sequence model + GBDT ranker."""
from __future__ import annotations
import pytest

from src.data.preprocessing import DataPreprocessor
from src.models.bert4rec import BERT4RecTrainer, BERT4RecConfig


@pytest.fixture
def processed(tmp_project):
    return DataPreprocessor(tmp_project).run(persist=False)


def test_bert4rec_smoke(processed):
    cfg = BERT4RecConfig(max_len=8, hidden_dim=16, n_heads=2, n_layers=1,
                         batch_size=4, epochs=2, lr=1e-2)
    tr = BERT4RecTrainer(cfg).fit(processed.train)
    history = list(processed.train[processed.train["user_id"] == 0]["job_id"])
    recs = tr.recommend(history, k=2)
    assert 0 <= len(recs) <= 2
    trained = set(processed.train["job_id"].unique())
    for jid, _ in recs:
        assert jid in trained
