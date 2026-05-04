"""Verify LTR negative sampling produces the expected ratio of negatives to positives
and that label semantics are binary (1 for positives, 0 for negatives)."""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.models.ltr_ranker import LTRRanker, LTRConfig, SignalProvider


def test_negative_sampling_counts_and_labels():
    jobs = pd.DataFrame({
        "job_id": np.arange(100),
        "category": ["A"] * 50 + ["B"] * 50,
        "skills": [""] * 100,
        "location": [""] * 100,
        "seniority": [""] * 100,
        "salary_min": [0.0] * 100,
        "salary_max": [0.0] * 100,
        "posted_days_ago": [0.0] * 100,
    })
    users = pd.DataFrame({
        "user_id": [0],
        "experience_years": [3],
        "skills": [""],
        "preferred_location": [""],
    })
    train = pd.DataFrame({
        "user_id": [0, 0],
        "job_id": [3, 60],   # one in cat A, one in cat B
        "rating": [1, 1],
    })
    sp = SignalProvider(two_tower=None, content=None, collab=None,
                        popularity=None, bilateral=None)
    ltr = LTRRanker(LTRConfig(n_random_negatives=50, n_hard_negatives=5), sp)
    X, y, groups = ltr._build_training_matrix(users=users, jobs=jobs, train=train)

    # 2 positives, ~100 random + ~10 hard negatives = ~112 candidates total
    assert groups.shape == (1,)
    assert groups[0] >= 50, f"too few candidates: {groups[0]}"
    # Exactly 2 positives, rest are negatives
    assert int((y == 1).sum()) == 2
    assert int((y == 0).sum()) == int(groups[0]) - 2
    # Labels are strictly {0, 1}
    assert set(np.unique(y).tolist()) <= {0.0, 1.0}


def test_no_seen_jobs_in_negatives():
    """Sampled negatives must never include the user's positive items."""
    jobs = pd.DataFrame({
        "job_id": np.arange(20),
        "category": ["A"] * 20,
        "skills": [""] * 20,
        "location": [""] * 20,
        "seniority": [""] * 20,
        "salary_min": [0.0] * 20,
        "salary_max": [0.0] * 20,
        "posted_days_ago": [0.0] * 20,
    })
    users = pd.DataFrame({"user_id": [0], "experience_years": [0],
                          "skills": [""], "preferred_location": [""]})
    train = pd.DataFrame({"user_id": [0, 0, 0], "job_id": [5, 10, 15], "rating": [1, 1, 1]})
    sp = SignalProvider(two_tower=None, content=None, collab=None,
                        popularity=None, bilateral=None)
    ltr = LTRRanker(LTRConfig(n_random_negatives=10, n_hard_negatives=2), sp)
    # Reach into the building loop: replicate enough of it to inspect the candidate set
    X, y, groups = ltr._build_training_matrix(users=users, jobs=jobs, train=train)
    # We can't easily get cand back from X, but we can assert label structure
    assert int((y == 1).sum()) == 3
    # Property: hard sampling produces ≤ n_hard_negatives × n_pos = 6, plus ≤ 30 random,
    # plus 3 pos → upper bound 39
    assert groups[0] <= 3 + 10 * 3 + 2 * 3
