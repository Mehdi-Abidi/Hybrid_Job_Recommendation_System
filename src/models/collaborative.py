"""Truncated-SVD collaborative filter over the sparse user-job rating matrix.
Reconstructed rating ≈ mean + U_k Σ_k V_k^T. Self-contained (no Surprise dep)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from src.utils.logging import get_logger

log = get_logger(__name__)


class CollaborativeRecommender:
    def __init__(self, n_factors: int = 100, n_epochs: int = 20, lr: float = 0.005, reg: float = 0.02, seed: int = 42):
        self.n_factors = n_factors
        # n_epochs/lr/reg retained for config compatibility; unused by svds.
        self.seed = seed
        self.U: np.ndarray | None = None          # (n_users, k)
        self.Vt: np.ndarray | None = None         # (k, n_jobs)
        self.user_bias: np.ndarray | None = None  # (n_users,)
        self.job_bias: np.ndarray | None = None   # (n_jobs,)
        self.global_mean: float = 0.0
        self._user_index: dict[int, int] = {}
        self._job_index: dict[int, int] = {}
        self._all_job_ids: np.ndarray | None = None
        self._trained_users: set[int] = set()
        self.rating_scale = (1.0, 5.0)

    def fit(self, train: pd.DataFrame, all_job_ids: np.ndarray,
            all_user_ids: np.ndarray | None = None) -> "CollaborativeRecommender":
        user_ids = np.asarray(all_user_ids) if all_user_ids is not None \
            else np.sort(train["user_id"].unique())
        job_ids = np.asarray(all_job_ids)
        self._user_index = {int(u): i for i, u in enumerate(user_ids)}
        self._job_index = {int(j): i for i, j in enumerate(job_ids)}
        self._all_job_ids = job_ids
        self._trained_users = set(int(u) for u in train["user_id"].unique())

        ratings = train["rating"].astype(np.float32).to_numpy()
        self.global_mean = float(ratings.mean())

        # Mean-centered sparse rating matrix for SVD.
        rows = train["user_id"].map(self._user_index).to_numpy()
        cols = train["job_id"].map(self._job_index).to_numpy()
        mask = (~pd.isna(rows)) & (~pd.isna(cols))
        rows, cols, ratings = rows[mask].astype(int), cols[mask].astype(int), ratings[mask]
        centered = ratings - self.global_mean
        R = csr_matrix((centered, (rows, cols)), shape=(len(user_ids), len(job_ids)))

        # svds requires k < min(R.shape); clamp.
        k = min(self.n_factors, min(R.shape) - 1)
        if k < 1:
            # Fallback: empty model -> predicts global mean.
            self.U = np.zeros((len(user_ids), 1), dtype=np.float32)
            self.Vt = np.zeros((1, len(job_ids)), dtype=np.float32)
        else:
            rng = np.random.default_rng(self.seed)
            v0 = rng.standard_normal(min(R.shape)).astype(np.float32)
            U, s, Vt = svds(R.astype(np.float32), k=k, v0=v0)
            # svds returns ascending singular values; no need to reorder for dot product.
            self.U = (U * s).astype(np.float32)
            self.Vt = Vt.astype(np.float32)

        # Per-user / per-job residual bias after factor reconstruction.
        pred_train = self._predict_batch(rows, cols)
        residual = ratings - (self.global_mean + pred_train)
        self.user_bias = np.zeros(len(user_ids), dtype=np.float32)
        self.job_bias = np.zeros(len(job_ids), dtype=np.float32)
        np.add.at(self.user_bias, rows, residual)
        counts_u = np.bincount(rows, minlength=len(user_ids)).astype(np.float32)
        self.user_bias /= np.maximum(counts_u, 1.0)

        log.info("CollaborativeFilter fit: %d users, %d jobs, k=%d, global_mean=%.3f",
                 len(user_ids), len(job_ids), k, self.global_mean)
        return self

    def _predict_batch(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return np.einsum("ij,ij->i", self.U[rows], self.Vt[:, cols].T).astype(np.float32)

    def predict(self, user_id: int, job_id: int) -> float:
        ui = self._user_index.get(int(user_id))
        ji = self._job_index.get(int(job_id))
        if ui is None or ji is None:
            return self.global_mean
        base = self.global_mean + float(self.U[ui] @ self.Vt[:, ji])
        if self.user_bias is not None:
            base += float(self.user_bias[ui])
        return float(np.clip(base, *self.rating_scale))

    def score_pairs(self, user_id: int, job_ids: list[int]) -> np.ndarray:
        ui = self._user_index.get(int(user_id))
        if ui is None:
            return np.full(len(job_ids), self.global_mean, dtype=np.float32)
        cols = np.array([self._job_index.get(int(j), -1) for j in job_ids])
        scores = np.full(len(job_ids), self.global_mean, dtype=np.float32)
        valid = cols >= 0
        scores[valid] = self.global_mean + self.U[ui] @ self.Vt[:, cols[valid]]
        if self.user_bias is not None:
            scores[valid] += self.user_bias[ui]
        return np.clip(scores, *self.rating_scale).astype(np.float32)

    def recommend(self, user_id: int, k: int = 10, exclude: set[int] | None = None) -> list[tuple[int, float]]:
        assert self._all_job_ids is not None
        exclude = exclude or set()
        job_ids = self._all_job_ids
        scores = self.score_pairs(user_id, list(job_ids))
        if exclude:
            for i, j in enumerate(job_ids):
                if int(j) in exclude:
                    scores[i] = -np.inf
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(job_ids[i]), float(scores[i])) for i in top]

    def knows_user(self, user_id: int) -> bool:
        return int(user_id) in self._trained_users

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.savez(path / "svd.npz",
                 U=self.U, Vt=self.Vt, user_bias=self.user_bias, job_bias=self.job_bias,
                 global_mean=np.float32(self.global_mean),
                 user_ids=np.array(list(self._user_index.keys()), dtype=np.int64),
                 job_ids=np.array(self._all_job_ids, dtype=np.int64),
                 trained_users=np.array(list(self._trained_users), dtype=np.int64))

    def load(self, path: Path) -> "CollaborativeRecommender":
        z = np.load(path / "svd.npz")
        self.U, self.Vt = z["U"], z["Vt"]
        self.user_bias, self.job_bias = z["user_bias"], z["job_bias"]
        self.global_mean = float(z["global_mean"])
        user_ids = z["user_ids"]
        self._user_index = {int(u): i for i, u in enumerate(user_ids)}
        self._all_job_ids = z["job_ids"]
        self._job_index = {int(j): i for i, j in enumerate(self._all_job_ids)}
        self._trained_users = set(int(u) for u in z["trained_users"])
        return self
