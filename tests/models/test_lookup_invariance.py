"""Verify dict lookups give identical results to the previous np.where pattern."""
from __future__ import annotations
import numpy as np
import pandas as pd


def test_dict_lookup_matches_np_where_for_user_indices():
    user_ids = np.array([10, 11, 12, 13])
    pairs = pd.DataFrame({"user_id": [11, 13, 11, 10], "job_id": [300, 100, 500, 200]})
    u_id_to_row = {int(u): i for i, u in enumerate(user_ids)}
    u_idx_dict = np.fromiter((u_id_to_row[int(u)] for u in pairs["user_id"]),
                             dtype=np.int64, count=len(pairs))
    u_idx_where = np.array([np.where(user_ids == u)[0][0] for u in pairs["user_id"]])
    assert np.array_equal(u_idx_dict, u_idx_where)


def test_dict_lookup_matches_np_where_for_job_indices_with_unknowns():
    job_ids = np.array([100, 200, 300, 400, 500])
    cands = [300, 999, 100, 200]  # 999 is unknown
    j_id_to_row = {int(j): i for i, j in enumerate(job_ids)}
    j_idx_dict = np.fromiter(
        (j_id_to_row.get(int(j), -1) for j in cands),
        dtype=np.int64, count=len(cands),
    )
    j_idx_where = np.array(
        [np.where(job_ids == j)[0][0] if j in job_ids else -1 for j in cands]
    )
    assert np.array_equal(j_idx_dict, j_idx_where)
