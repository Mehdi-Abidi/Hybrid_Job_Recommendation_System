"""Content-based recommender: cosine similarity between user profile and job embeddings."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from src.features.text_features import EmbeddingFeaturizer, job_text
from src.features.user_profile import UserProfileBuilder
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ContentArtifacts:
    job_embeddings: np.ndarray    # (n_jobs, dim), L2-normalized
    user_profiles: np.ndarray     # (n_users, dim), L2-normalized
    job_ids: np.ndarray
    user_ids: np.ndarray


class ContentBasedRecommender:
    def __init__(self, embedder: EmbeddingFeaturizer | None = None):
        self.embedder = embedder or EmbeddingFeaturizer()
        self.profile_builder = UserProfileBuilder(self.embedder)
        self.artifacts: ContentArtifacts | None = None
        self._job_index: dict[int, int] = {}
        self._user_index: dict[int, int] = {}

    def fit(self, jobs: pd.DataFrame, users: pd.DataFrame, train: pd.DataFrame) -> "ContentBasedRecommender":
        job_texts = [job_text(r) for _, r in jobs.iterrows()]
        job_emb = self.embedder.encode(job_texts, normalize=True)
        job_ids = jobs["job_id"].to_numpy()
        self._job_index = {int(j): i for i, j in enumerate(job_ids)}
        profiles = self.profile_builder.build(users, jobs, train, job_emb, self._job_index)
        user_ids = users["user_id"].to_numpy()
        self._user_index = {int(u): i for i, u in enumerate(user_ids)}
        self.artifacts = ContentArtifacts(job_emb, profiles, job_ids, user_ids)
        log.info("ContentBased fit: %d jobs, %d users, dim=%d", len(job_ids), len(user_ids), job_emb.shape[1])
        return self

    # Rank jobs for a known user by cosine similarity.
    def recommend(self, user_id: int, k: int = 10, exclude: set[int] | None = None) -> list[tuple[int, float]]:
        assert self.artifacts is not None
        i = self._user_index[int(user_id)]
        scores = self.artifacts.job_embeddings @ self.artifacts.user_profiles[i]
        return self._top_k(scores, k, exclude)

    # Cold-start: rank jobs by similarity to freshly-encoded resume+skills.
    def recommend_for_new_user(self, resume: str, skills: str, k: int = 10) -> list[tuple[int, float]]:
        assert self.artifacts is not None
        vec = self.profile_builder.build_for_new_user(resume, skills)
        scores = self.artifacts.job_embeddings @ vec
        return self._top_k(scores, k, None)

    # Job-to-job similarity.
    def similar_jobs(self, job_id: int, k: int = 10) -> list[tuple[int, float]]:
        assert self.artifacts is not None
        j = self._job_index[int(job_id)]
        scores = self.artifacts.job_embeddings @ self.artifacts.job_embeddings[j]
        scores[j] = -np.inf
        return self._top_k(scores, k, None)

    # Score any (user, job) pair — used by the hybrid combiner.
    def score_pairs(self, user_id: int, job_ids: list[int]) -> np.ndarray:
        assert self.artifacts is not None
        i = self._user_index[int(user_id)]
        cols = np.array([self._job_index[int(j)] for j in job_ids])
        return self.artifacts.job_embeddings[cols] @ self.artifacts.user_profiles[i]

    def _top_k(self, scores: np.ndarray, k: int, exclude: set[int] | None) -> list[tuple[int, float]]:
        if exclude:
            for jid in exclude:
                idx = self._job_index.get(int(jid))
                if idx is not None:
                    scores[idx] = -np.inf
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        job_ids = self.artifacts.job_ids
        return [(int(job_ids[i]), float(scores[i])) for i in top]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "job_embeddings.npy", self.artifacts.job_embeddings)
        np.save(path / "user_profiles.npy", self.artifacts.user_profiles)
        np.save(path / "job_ids.npy", self.artifacts.job_ids)
        np.save(path / "user_ids.npy", self.artifacts.user_ids)

    def load(self, path: Path) -> "ContentBasedRecommender":
        job_emb = np.load(path / "job_embeddings.npy")
        profiles = np.load(path / "user_profiles.npy")
        job_ids = np.load(path / "job_ids.npy")
        user_ids = np.load(path / "user_ids.npy")
        self._job_index = {int(j): i for i, j in enumerate(job_ids)}
        self._user_index = {int(u): i for i, u in enumerate(user_ids)}
        self.artifacts = ContentArtifacts(job_emb, profiles, job_ids, user_ids)
        return self
