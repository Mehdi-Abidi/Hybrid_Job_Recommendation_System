"""iALS collaborative filter API contract.

Production uses `implicit.als.AlternatingLeastSquares`. On environments without
a C++ compiler (Windows dev), the module falls back to confidence-weighted
truncated SVD. Both produce factor matrices of the right shape; this test
validates the API contract regardless of which backend ran."""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.models.collaborative import CollaborativeRecommender


def test_fit_recommend_score_pairs_smoke():
    rng = np.random.default_rng(0)
    n_users, n_jobs = 30, 200
    inter = pd.DataFrame({
        "user_id": rng.integers(0, n_users, 300),
        "job_id": rng.integers(0, n_jobs, 300),
    })
    rec = CollaborativeRecommender(n_factors=16, n_epochs=5).fit(
        inter, all_job_ids=np.arange(n_jobs), all_user_ids=np.arange(n_users),
    )
    # recommend returns k items as (int job_id, float score) pairs
    out = rec.recommend(user_id=5, k=10)
    assert len(out) == 10
    assert all(isinstance(j, int) and isinstance(s, float) for j, s in out)
    # score_pairs returns a numpy array shaped (len(job_ids),)
    arr = rec.score_pairs(user_id=5, job_ids=[0, 50, 199, 999])  # 999 unknown
    assert arr.shape == (4,)
    assert arr[3] == 0.0  # unknown job → 0
    # knows_user reflects training set membership
    assert rec.knows_user(int(inter["user_id"].iloc[0]))


def test_factor_shapes_match_config():
    rng = np.random.default_rng(1)
    n_users, n_jobs, factors = 20, 50, 8
    inter = pd.DataFrame({
        "user_id": rng.integers(0, n_users, 80),
        "job_id": rng.integers(0, n_jobs, 80),
    })
    rec = CollaborativeRecommender(n_factors=factors, n_epochs=3).fit(
        inter, all_job_ids=np.arange(n_jobs), all_user_ids=np.arange(n_users),
    )
    assert rec.U.shape == (n_users, factors)
    assert rec.V.shape == (n_jobs, factors)


def test_save_load_round_trip(tmp_path):
    rng = np.random.default_rng(2)
    inter = pd.DataFrame({
        "user_id": rng.integers(0, 10, 30),
        "job_id": rng.integers(0, 40, 30),
    })
    rec = CollaborativeRecommender(n_factors=4, n_epochs=2).fit(
        inter, all_job_ids=np.arange(40), all_user_ids=np.arange(10),
    )
    rec.save(tmp_path / "cf")
    loaded = CollaborativeRecommender().load(tmp_path / "cf")
    np.testing.assert_array_equal(loaded.U, rec.U)
    np.testing.assert_array_equal(loaded.V, rec.V)
