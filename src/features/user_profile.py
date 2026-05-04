"""Build dual user profile vectors: long-term (all-time history) + short-term (recent).

Inspired by PinnerSAGE (KDD 2020) and LinkedIn LiGNN — single-vector user towers
regress to the mean of all-time activity and underweight recent intent. Producing
both vectors and scoring as 0.5 * (cos(long, job) + cos(short, job)) consistently
adds 5–10% engagement lift in published A/Bs.

Cold-start path: when a user has no interactions, both long and short collapse
to the resume-text encoding."""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.features.text_features import EmbeddingFeaturizer, user_text


class UserProfileBuilder:
    """Warm users: rating-weighted mean of interacted job embeddings (long-term)
    + mean of last N interactions by recency (short-term).
    Cold users: encode resume text directly into both slots."""

    def __init__(self, embedder: EmbeddingFeaturizer, short_term_n: int = 5):
        self.embedder = embedder
        self.short_term_n = short_term_n

    # Build (n_users, 2*dim) profile matrix: [long_term_normed | short_term_normed].
    def build(self, users: pd.DataFrame, jobs: pd.DataFrame, train: pd.DataFrame,
              job_embeddings: np.ndarray, job_index: dict[int, int]) -> np.ndarray:
        n_users, dim = len(users), job_embeddings.shape[1]
        long = np.zeros((n_users, dim), dtype=np.float32)
        short = np.zeros((n_users, dim), dtype=np.float32)
        user_row = {uid: i for i, uid in enumerate(users["user_id"].to_numpy())}
        cold_uids: list[int] = []

        has_time = "timestamp_days_ago" in train.columns
        has_rating = "rating" in train.columns
        grouped = train.groupby("user_id")
        interacted = set(grouped.groups.keys())

        for uid, i in user_row.items():
            if uid not in interacted:
                cold_uids.append(uid)
                continue
            g = grouped.get_group(uid)
            cols = g["job_id"].map(job_index).to_numpy()
            mask = ~pd.isna(cols)
            cols = cols[mask].astype(int)
            if len(cols) == 0:
                cold_uids.append(uid)
                continue
            ratings = (g["rating"].to_numpy(dtype=np.float32)[mask]
                       if has_rating else np.ones(len(cols), dtype=np.float32))
            # Long-term: rating-weighted mean of all interacted job vectors.
            w = ratings / max(ratings.sum(), 1e-12)
            long[i] = (job_embeddings[cols] * w[:, None]).sum(axis=0)
            # Short-term: mean of last N interactions by recency. timestamp_days_ago:
            # smaller = more recent → take the N smallest.
            if has_time:
                ts = g["timestamp_days_ago"].to_numpy()[mask]
                order = np.argsort(ts)[: self.short_term_n]
                recent_cols = cols[order]
            else:
                recent_cols = cols[-self.short_term_n :]
            short[i] = job_embeddings[recent_cols].mean(axis=0)

        if cold_uids:
            idx = [user_row[u] for u in cold_uids]
            cold_vecs = self.embedder.encode(
                [user_text(users.iloc[i]) for i in idx], normalize=True,
            )
            long[idx] = cold_vecs
            short[idx] = cold_vecs

        long /= (np.linalg.norm(long, axis=1, keepdims=True) + 1e-12)
        short /= (np.linalg.norm(short, axis=1, keepdims=True) + 1e-12)
        return np.concatenate([long, short], axis=1).astype(np.float32)

    # Cold-start: build a single (2*dim,) profile from resume + skills.
    # Optional `ner` and `ontology` augment the skill set before encoding (Task 7).
    def build_for_new_user(self, resume: str, skills: str, *,
                           ner=None, ontology=None) -> np.ndarray:
        from src.features.structured_features import parse_skills
        skill_set = parse_skills(skills)
        if ner is not None:
            try:
                extracted = ner.extract_skills(resume)
                skill_set = skill_set | {s.strip().lower() for s in extracted if s}
            except Exception:
                pass
        if ontology is not None:
            try:
                skill_set = ontology.expand(skill_set)
            except Exception:
                pass
        skill_str = ",".join(sorted(skill_set))
        text = f"{resume} Skills: {skill_str}"
        vec = self.embedder.encode([text], normalize=True)[0]
        # Cold users have no history — short = long.
        return np.concatenate([vec, vec]).astype(np.float32)
