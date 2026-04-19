"""Tests for metrics and Evaluator."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import DataPreprocessor
from src.evaluation.metrics import (
    precision_at_k, recall_at_k, ndcg_at_k, average_precision,
    coverage, intra_list_diversity,
)
from src.evaluation.evaluator import Evaluator


# Precision/recall on a hand-constructed case.
def test_precision_recall_hand_computed():
    recs = [1, 2, 3, 4, 5]
    rel = {2, 5, 9}
    assert precision_at_k(recs, rel, 5) == pytest.approx(2 / 5)
    assert recall_at_k(recs, rel, 5) == pytest.approx(2 / 3)
    assert precision_at_k(recs, rel, 1) == 0.0  # item 1 not relevant


# NDCG ideal ordering yields 1.0.
def test_ndcg_perfect():
    recs = [1, 2, 3]
    rel = {1, 2, 3}
    assert ndcg_at_k(recs, rel, 3) == pytest.approx(1.0)


# NDCG bad ordering is < 1.0 but > 0.
def test_ndcg_suboptimal():
    recs = [9, 8, 1]  # 1 is relevant, late in list
    rel = {1}
    v = ndcg_at_k(recs, rel, 3)
    assert 0 < v < 1


# MAP on a classic two-hit example.
def test_map():
    recs = [1, 2, 3, 4, 5]
    rel = {1, 3}
    # hits at positions 1 and 3 → AP = (1/1 + 2/3) / 2
    assert average_precision(recs, rel) == pytest.approx((1.0 + 2 / 3) / 2)


# Coverage counts the union of recommended items across users.
def test_coverage():
    all_recs = [[1, 2], [2, 3], [3, 4]]
    assert coverage(all_recs, n_items=10) == pytest.approx(4 / 10)


# Intra-list diversity = 0 when all embeddings identical, higher when orthogonal.
def test_diversity_identity_vs_orthogonal():
    emb = np.eye(3, dtype=np.float32)
    id_to_row = {0: 0, 1: 1, 2: 2}
    assert intra_list_diversity([0, 1, 2], emb, id_to_row) == pytest.approx(1.0)
    emb2 = np.ones((3, 3), dtype=np.float32)
    assert intra_list_diversity([0, 1, 2], emb2, id_to_row) == pytest.approx(0.0, abs=1e-4)


# Evaluator end-to-end on a trivial oracle recommender (should hit perfect precision@1).
def test_evaluator_oracle(tmp_project):
    data = DataPreprocessor(tmp_project).run(persist=False)
    # Oracle: for each user, place a true test job at position 1.
    test_by_user = {int(u): list(g["job_id"]) for u, g in data.test.groupby("user_id")}
    all_jobs = list(data.jobs["job_id"])

    def oracle(user_id: int, k: int):
        true = test_by_user.get(user_id, [])
        others = [j for j in all_jobs if j not in true]
        ranked = [(j, 1.0) for j in true] + [(j, 0.0) for j in others]
        return ranked[:k]

    ev = Evaluator(data.test, k_values=[1, 2], jobs=data.jobs, positive_rating_threshold=1)
    rep = ev.evaluate(oracle, "oracle")
    # Oracle hits at position 1 for users with ≥1 relevant test item.
    assert rep.precision[1] == pytest.approx(1.0)
    assert rep.ndcg[1] == pytest.approx(1.0)


# Evaluator compare_models produces a DataFrame with one row per model.
def test_evaluator_compare_models(tmp_project):
    data = DataPreprocessor(tmp_project).run(persist=False)
    all_jobs = list(data.jobs["job_id"])
    models = {
        "random_a": lambda u, k: [(j, 0.5) for j in all_jobs[:k]],
        "random_b": lambda u, k: [(j, 0.5) for j in all_jobs[::-1][:k]],
    }
    ev = Evaluator(data.test, k_values=[2], jobs=data.jobs, positive_rating_threshold=1)
    df = ev.compare_models(models)
    assert set(df["model"]) == {"random_a", "random_b"}
    assert "precision@2" in df.columns


# Cold-start analysis returns one row per bucket.
def test_cold_start_analysis(tmp_project):
    data = DataPreprocessor(tmp_project).run(persist=False)
    all_jobs = list(data.jobs["job_id"])
    ev = Evaluator(data.test, k_values=[2], jobs=data.jobs, positive_rating_threshold=1)
    df = ev.cold_start_analysis(lambda u, k: [(j, 0.5) for j in all_jobs[:k]], "const",
                                train=data.train, buckets=((0, 0), (1, 10)))
    assert len(df) == 2
    assert set(df["bucket"]) == {"0-0", "1-10"}
