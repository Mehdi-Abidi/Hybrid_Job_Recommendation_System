"""Tests for reciprocal recommendation: JobToUserTower, BilateralScorer, and the
SIGKDD'24 bilateral metrics. Stays offline by reusing the FakeEmbedder pattern."""
from __future__ import annotations
import numpy as np
import pytest

from src.data.preprocessing import DataPreprocessor
from src.models.two_tower import TwoTowerTrainer
from src.models.reciprocal import JobToUserTower, BilateralScorer, ReciprocalConfig
from src.evaluation.metrics import (
    bilateral_coverage, balanced_ranking_ratio, two_sided_ndcg,
)
from tests.test_neural_retrieval import FakeEmbedder


@pytest.fixture
def processed(tmp_project):
    return DataPreprocessor(tmp_project).run(persist=False)


@pytest.fixture
def trained_forward(processed):
    cfg = {"embedding_dim": 8, "hidden_dims": [16], "dropout": 0.0,
           "batch_size": 8, "epochs": 2, "lr": 1e-2}
    trainer = TwoTowerTrainer(cfg, text_embedder=FakeEmbedder(dim=8), seed=0)
    trainer.build_features(processed.jobs, processed.users)
    trainer.train(processed.train, epochs=2, batch_size=8, lr=1e-2)
    return trainer


# JobToUserTower trains without error and produces L2-normalized embeddings on both sides.
def test_inverse_tower_trains_and_normalizes(trained_forward, processed):
    rcfg = ReciprocalConfig(embedding_dim=8, hidden_dims=(16,), dropout=0.0,
                            epochs=2, batch_size=8, lr=1e-2, seed=0)
    inv = JobToUserTower(trained_forward.artifacts, rcfg).fit(processed.train)
    ue = inv.user_embeddings()
    je = inv.job_embeddings()
    assert ue.shape == (len(processed.users), 8)
    assert je.shape == (len(processed.jobs), 8)
    assert np.allclose(np.linalg.norm(ue, axis=1), 1.0, atol=1e-4)
    assert np.allclose(np.linalg.norm(je, axis=1), 1.0, atol=1e-4)


# score_pairs returns dot products in [-1, 1] for L2-normalized embeddings, and
# unknown ids resolve to 0.0 rather than raising.
def test_inverse_tower_score_pairs(trained_forward, processed):
    rcfg = ReciprocalConfig(embedding_dim=8, hidden_dims=(16,), dropout=0.0,
                            epochs=1, batch_size=8, lr=1e-2, seed=0)
    inv = JobToUserTower(trained_forward.artifacts, rcfg).fit(processed.train)
    user_id = int(processed.users["user_id"].iloc[0])
    job_ids = processed.jobs["job_id"].astype(int).tolist() + [99999]  # last is unknown
    scores = inv.score_pairs(user_id, job_ids)
    assert scores.shape == (len(job_ids),)
    assert (scores[:-1] >= -1.0 - 1e-5).all() and (scores[:-1] <= 1.0 + 1e-5).all()
    assert scores[-1] == 0.0


# BilateralScorer.score returns (fwd, inv, bilateral). Bilateral = sigmoid(fwd)*sigmoid(inv),
# strictly in (0, 1), and in (0.0, 0.25] when both sides are <= 0 (ablation sanity).
def test_bilateral_scorer_range_and_product(trained_forward, processed):
    rcfg = ReciprocalConfig(embedding_dim=8, hidden_dims=(16,), dropout=0.0,
                            epochs=1, batch_size=8, lr=1e-2, seed=0)
    inv = JobToUserTower(trained_forward.artifacts, rcfg).fit(processed.train)
    bilat = BilateralScorer(forward_tower=trained_forward, inverse_tower=inv)
    user_id = int(processed.users["user_id"].iloc[0])
    job_ids = processed.jobs["job_id"].astype(int).tolist()
    fwd, ji, b = bilat.score(user_id, job_ids)
    assert fwd.shape == ji.shape == b.shape == (len(job_ids),)
    assert (b > 0).all() and (b < 1).all()
    # Bilateral must equal the elementwise product of the two sigmoids.
    expected = (1.0 / (1.0 + np.exp(-fwd))) * (1.0 / (1.0 + np.exp(-ji)))
    assert np.allclose(b, expected, atol=1e-5)


# Empty job_ids returns three empty arrays without error.
def test_bilateral_scorer_empty(trained_forward, processed):
    rcfg = ReciprocalConfig(embedding_dim=8, hidden_dims=(16,), dropout=0.0,
                            epochs=1, batch_size=8, lr=1e-2, seed=0)
    inv = JobToUserTower(trained_forward.artifacts, rcfg).fit(processed.train)
    bilat = BilateralScorer(forward_tower=trained_forward, inverse_tower=inv)
    fwd, ji, b = bilat.score(0, [])
    assert fwd.size == 0 and ji.size == 0 and b.size == 0


# bilateral_coverage on a hand-crafted set: 4 items where 2 are above-median on both sides.
def test_bilateral_coverage_known_values():
    rec = [10, 20, 30, 40]
    user_scores = {10: 0.9, 20: 0.8, 30: 0.2, 40: 0.1}
    job_scores  = {10: 0.9, 20: 0.1, 30: 0.8, 40: 0.2}
    # medians: u_med = 0.5, v_med = 0.5 → only id 10 (0.9, 0.9) is >= both medians.
    # However >= median includes the median itself; with 4 items median = (0.8+0.2)/2 = 0.5
    # for u and (0.8+0.2)/2 = 0.5 for v. Items >= 0.5 on u: {10,20}; on v: {10,30}. Both: {10}.
    cov = bilateral_coverage(rec, user_scores, job_scores)
    assert cov == pytest.approx(0.25)


def test_bilateral_coverage_empty_returns_zero():
    assert bilateral_coverage([], {}, {}) == 0.0


# balanced_ranking_ratio: 1.0 for equal sides, < 1 for asymmetric, 0 for either side undefined.
def test_balanced_ranking_ratio():
    assert balanced_ranking_ratio(0.6, 0.6) == pytest.approx(1.0)
    assert balanced_ranking_ratio(0.8, 0.4) == pytest.approx(0.5)
    assert balanced_ranking_ratio(0.0, 0.5) == 0.0
    assert balanced_ranking_ratio(0.5, 0.0) == 0.0


# two_sided_ndcg: geometric mean. Punishes zero on either side.
def test_two_sided_ndcg():
    assert two_sided_ndcg(0.9, 0.0) == 0.0
    assert two_sided_ndcg(0.4, 0.9) == pytest.approx(np.sqrt(0.36))
    assert two_sided_ndcg(0.5, 0.5) == pytest.approx(0.5)


# Save/load round-trips the inverse tower and produces identical embeddings.
def test_inverse_tower_save_load_roundtrip(tmp_path, trained_forward, processed):
    rcfg = ReciprocalConfig(embedding_dim=8, hidden_dims=(16,), dropout=0.0,
                            epochs=1, batch_size=8, lr=1e-2, seed=0)
    inv = JobToUserTower(trained_forward.artifacts, rcfg).fit(processed.train)
    inv.save(tmp_path)
    inv2 = JobToUserTower(trained_forward.artifacts, rcfg).load(tmp_path)
    e1, e2 = inv.user_embeddings(), inv2.user_embeddings()
    assert np.allclose(e1, e2, atol=1e-5)


# RankingSignals with reciprocal fields produces 14-dim feature rows; without them, 14-dim
# but trailing 3 columns are zero — verifies the LTR feature builder integration.
def test_ranking_features_with_and_without_bilateral(trained_forward, processed):
    from src.features.ranking_features import RankingSignals, build_ranking_features, FEATURE_NAMES
    user_id = int(processed.users["user_id"].iloc[0])
    cand = processed.jobs["job_id"].astype(int).tolist()
    n = len(cand)

    # Without bilateral: trailing 3 columns must be 0.
    sigs = RankingSignals(
        two_tower=np.full(n, 0.5, dtype=np.float32),
        content=np.full(n, 0.4, dtype=np.float32),
        collab=np.full(n, 3.0, dtype=np.float32),
        popularity=np.full(n, 0.1, dtype=np.float32),
    )
    feats = build_ranking_features(user_id, cand, processed.users, processed.jobs, sigs)
    assert feats.shape == (n, len(FEATURE_NAMES))
    assert (feats[:, -3:] == 0).all()

    # With bilateral: trailing 3 columns are the supplied values.
    s_uj = np.linspace(-0.5, 0.5, n).astype(np.float32)
    s_ju = np.linspace(0.1, 0.6, n).astype(np.float32)
    bilat = (1 / (1 + np.exp(-s_uj))) * (1 / (1 + np.exp(-s_ju)))
    sigs2 = RankingSignals(
        two_tower=sigs.two_tower, content=sigs.content, collab=sigs.collab, popularity=sigs.popularity,
        s_user_to_job=s_uj, s_job_to_user=s_ju, bilateral=bilat.astype(np.float32),
    )
    feats2 = build_ranking_features(user_id, cand, processed.users, processed.jobs, sigs2)
    assert np.allclose(feats2[:, -3], s_uj, atol=1e-5)
    assert np.allclose(feats2[:, -2], s_ju, atol=1e-5)
    assert np.allclose(feats2[:, -1], bilat, atol=1e-5)
