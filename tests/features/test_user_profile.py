"""User profile is now (n_users, 2*dim): [long-term | short-term]."""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.features.user_profile import UserProfileBuilder


class _StubEmbedder:
    """Deterministic stand-in for sentence-transformers — encodes by hashing tokens."""
    def encode(self, texts, normalize=True):
        out = np.zeros((len(texts), 8), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in str(t).split():
                out[i, hash(tok) % 8] += 1.0
            n = np.linalg.norm(out[i])
            if normalize and n > 0:
                out[i] /= n
        return out


def test_profile_shape_is_double_dim():
    job_emb = np.random.default_rng(0).standard_normal((10, 8)).astype(np.float32)
    job_emb /= np.linalg.norm(job_emb, axis=1, keepdims=True)
    job_index = {j: j for j in range(10)}
    users = pd.DataFrame({"user_id": [0, 1], "skills": ["", ""], "resume_text": ["resume", "resume"]})
    train = pd.DataFrame({
        "user_id": [0, 0, 0, 1],
        "job_id": [3, 5, 7, 2],
        "rating": [1, 1, 1, 1],
        "timestamp_days_ago": [10, 5, 1, 2],
    })
    profiles = UserProfileBuilder(_StubEmbedder()).build(
        users, jobs=pd.DataFrame(), train=train,
        job_embeddings=job_emb, job_index=job_index,
    )
    assert profiles.shape == (2, 16)  # 2 users × (8 long + 8 short)


def test_short_term_uses_most_recent_interactions():
    """Short-term half should equal mean of last-N (smallest timestamp_days_ago) job vecs."""
    job_emb = np.eye(10, dtype=np.float32)
    job_index = {j: j for j in range(10)}
    users = pd.DataFrame({"user_id": [0], "skills": [""], "resume_text": ["x"]})
    train = pd.DataFrame({
        "user_id": [0, 0, 0, 0, 0],
        "job_id": [0, 1, 2, 3, 4],
        "rating": [1, 1, 1, 1, 1],
        # Job 4 is most recent (1 day), then 3, 2, 1, 0.
        "timestamp_days_ago": [50, 40, 30, 20, 1],
    })
    profiles = UserProfileBuilder(_StubEmbedder(), short_term_n=2).build(
        users, jobs=pd.DataFrame(), train=train,
        job_embeddings=job_emb, job_index=job_index,
    )
    dim = job_emb.shape[1]
    short = profiles[0, dim:]  # second half
    # Last 2 most-recent jobs are 4 and 3 — mean of e_4 and e_3 then normalized.
    expected = (np.eye(10)[4] + np.eye(10)[3]) / 2
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(short, expected, atol=1e-5)


def test_cold_user_long_equals_short():
    """A user with no interactions: both halves collapse to the resume encoding."""
    job_emb = np.random.default_rng(0).standard_normal((10, 8)).astype(np.float32)
    job_index = {j: j for j in range(10)}
    users = pd.DataFrame({"user_id": [0, 1], "skills": ["", ""], "resume_text": ["a b", "x y"]})
    train = pd.DataFrame({"user_id": [0], "job_id": [0], "rating": [1], "timestamp_days_ago": [1]})
    profiles = UserProfileBuilder(_StubEmbedder()).build(
        users, jobs=pd.DataFrame(), train=train,
        job_embeddings=job_emb, job_index=job_index,
    )
    cold = profiles[1]
    long, short = cold[:8], cold[8:]
    np.testing.assert_allclose(long, short)
