"""Build a dense user profile vector from interaction history or resume text."""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.features.text_features import EmbeddingFeaturizer, user_text


class UserProfileBuilder:
    """Warm users: rating-weighted mean of interacted job embeddings.
    Cold users: encode resume text directly."""

    def __init__(self, embedder: EmbeddingFeaturizer):
        self.embedder = embedder

    # Build profile vectors for every user in `users`.
    def build(self, users: pd.DataFrame, jobs: pd.DataFrame, train: pd.DataFrame,
              job_embeddings: np.ndarray, job_index: dict[int, int]) -> np.ndarray:
        n_users, dim = len(users), job_embeddings.shape[1]
        profiles = np.zeros((n_users, dim), dtype=np.float32)
        user_row = {uid: i for i, uid in enumerate(users["user_id"].to_numpy())}
        cold_uids: list[int] = []

        grouped = train.groupby("user_id")
        interacted = set(grouped.groups.keys())

        for uid, i in user_row.items():
            if uid not in interacted:
                cold_uids.append(uid)
                continue
            g = grouped.get_group(uid)
            ratings = g["rating"].to_numpy(dtype=np.float32)
            cols = g["job_id"].map(job_index).to_numpy()
            mask = ~pd.isna(cols)
            cols = cols[mask].astype(int)
            ratings = ratings[mask]
            if len(cols) == 0:
                cold_uids.append(uid)
                continue
            vecs = job_embeddings[cols]
            w = ratings / ratings.sum()
            profiles[i] = (vecs * w[:, None]).sum(axis=0)

        if cold_uids:
            cold_idx = [user_row[u] for u in cold_uids]
            cold_texts = [user_text(users.iloc[i]) for i in cold_idx]
            profiles[cold_idx] = self.embedder.encode(cold_texts, normalize=True)

        norms = np.linalg.norm(profiles, axis=1, keepdims=True) + 1e-12
        return (profiles / norms).astype(np.float32)

    # Build a single profile for a brand-new user with resume + skills.
    def build_for_new_user(self, resume: str, skills: str) -> np.ndarray:
        text = f"{resume} Skills: {skills}"
        vec = self.embedder.encode([text], normalize=True)[0]
        return vec
