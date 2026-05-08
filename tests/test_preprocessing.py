"""Tests for src.data.preprocessing."""
from __future__ import annotations
import pandas as pd
from src.data.preprocessing import DataPreprocessor


# Full pipeline produces clean outputs with correct shapes and no leakage.
def test_full_pipeline(tmp_project):
    prep = DataPreprocessor(tmp_project)
    data = prep.run(persist=True)

    assert len(data.jobs) == 3
    assert len(data.users) == 2
    assert len(data.train) + len(data.test) == 5   # original interaction count preserved
    # Train and test must be disjoint on (user_id, job_id)
    pairs_train = set(zip(data.train["user_id"], data.train["job_id"]))
    pairs_test = set(zip(data.test["user_id"], data.test["job_id"]))
    assert pairs_train.isdisjoint(pairs_test)


# Rating mapping must follow config (view=1, save=3, apply=5).
def test_ratings_assigned(tmp_project):
    data = DataPreprocessor(tmp_project).run(persist=False)
    all_ix = pd.concat([data.train, data.test])
    action_to_rating = all_ix.groupby("action")["rating"].first().to_dict()
    assert action_to_rating["view"] == 1
    assert action_to_rating["save"] == 3
    assert action_to_rating["apply"] == 5


# Interaction matrix shape matches (n_users, n_jobs) and nnz ≤ train size.
def test_interaction_matrix_shape(tmp_project):
    data = DataPreprocessor(tmp_project).run(persist=False)
    assert data.interaction_matrix.shape == (len(data.users), len(data.jobs))
    assert data.interaction_matrix.nnz <= len(data.train)


# save/load roundtrip preserves data integrity.
def test_save_load_roundtrip(tmp_project):
    prep = DataPreprocessor(tmp_project)
    original = prep.run(persist=True)
    loaded = prep.load()
    pd.testing.assert_frame_equal(original.jobs, loaded.jobs)
    pd.testing.assert_frame_equal(original.users, loaded.users)
    assert original.interaction_matrix.nnz == loaded.interaction_matrix.nnz
    assert original.user_index == loaded.user_index


# Duplicate (user, job) pairs collapse to the strongest action.
def test_duplicate_interactions_collapse(tmp_project):
    raw = pd.read_csv(tmp_project.path("raw") / "interactions.csv")
    # Add a duplicate with a weaker action than the existing 'apply'
    extra = pd.DataFrame([{"user_id": 0, "job_id": 0, "action": "view", "timestamp_days_ago": 0}])
    combined = pd.concat([raw, extra], ignore_index=True)
    combined.to_csv(tmp_project.path("raw") / "interactions.csv", index=False)

    data = DataPreprocessor(tmp_project).run(persist=False)
    all_ix = pd.concat([data.train, data.test])
    row = all_ix[(all_ix["user_id"] == 0) & (all_ix["job_id"] == 0)]
    assert len(row) == 1                  # deduped
    assert row.iloc[0]["rating"] == 5     # kept stronger 'apply'


# Users with only one interaction go entirely to train (not orphaned into test).
def test_single_interaction_user_goes_to_train(tmp_project):
    # user 1 has 2 interactions in fixture; force to 1 by overwriting.
    raw = pd.read_csv(tmp_project.path("raw") / "interactions.csv")
    raw = raw[~((raw["user_id"] == 1) & (raw["job_id"] == 0))]
    raw.to_csv(tmp_project.path("raw") / "interactions.csv", index=False)

    data = DataPreprocessor(tmp_project).run(persist=False)
    u1_test = data.test[data.test["user_id"] == 1]
    assert len(u1_test) == 0
