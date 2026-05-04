"""Implicit ALS collaborative filter over the binary user-job interaction matrix.

Replaces the prior SVD-on-synthetic-ratings approach. Treats every observed
interaction as a positive signal with confidence scaled by `alpha` (Hu/Koren/
Volinsky 2008). Score for unseen (u,j) is the dot product of learned user
and item factors — values are unbounded reals, meaningful only as a ranking.

Backed by `implicit.als.AlternatingLeastSquares`. The compat shims (`predict`,
`global_mean`) preserve the call surface used by the legacy hybrid combiner."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.utils.logging import get_logger

log = get_logger(__name__)


class CollaborativeRecommender:
    """iALS on (user × job) binary interactions."""

    def __init__(self, n_factors: int = 64, n_epochs: int = 20,
                 reg: float = 0.01, alpha: float = 40.0, seed: int = 42, **_kwargs):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.reg = reg
        self.alpha = alpha
        self.seed = seed
        self.U: np.ndarray | None = None        # (n_users, n_factors)
        self.V: np.ndarray | None = None        # (n_jobs, n_factors)
        self._user_index: dict[int, int] = {}
        self._job_index: dict[int, int] = {}
        self._all_job_ids: np.ndarray | None = None
        self._trained_users: set[int] = set()

    def fit(self, train: pd.DataFrame, all_job_ids: np.ndarray,
            all_user_ids: np.ndarray | None = None) -> "CollaborativeRecommender":
        user_ids = np.asarray(all_user_ids) if all_user_ids is not None \
            else np.sort(train["user_id"].unique())
        job_ids = np.asarray(all_job_ids)
        self._user_index = {int(u): i for i, u in enumerate(user_ids)}
        self._job_index = {int(j): i for i, j in enumerate(job_ids)}
        self._all_job_ids = job_ids
        self._trained_users = set(int(u) for u in train["user_id"].unique())

        rows = train["user_id"].map(self._user_index).to_numpy()
        cols = train["job_id"].map(self._job_index).to_numpy()
        mask = (~pd.isna(rows)) & (~pd.isna(cols))
        rows, cols = rows[mask].astype(int), cols[mask].astype(int)
        # Binary positives — confidence handled by iALS through `alpha`.
        data = np.ones(len(rows), dtype=np.float32)
        ui_matrix = csr_matrix((data, (rows, cols)),
                               shape=(len(user_ids), len(job_ids)))

        try:
            from implicit.als import AlternatingLeastSquares
            model = AlternatingLeastSquares(
                factors=self.n_factors, regularization=self.reg,
                alpha=self.alpha, iterations=self.n_epochs,
                random_state=self.seed, use_gpu=False,
                calculate_training_loss=False,
            )
            model.fit(ui_matrix, show_progress=False)
            self.U = np.asarray(model.user_factors, dtype=np.float32)
            self.V = np.asarray(model.item_factors, dtype=np.float32)
            log.info("iALS fit: %d users, %d jobs, factors=%d, alpha=%.1f",
                     len(user_ids), len(job_ids), self.n_factors, self.alpha)
        except ImportError:
            # Fallback for environments where `implicit` is unavailable (e.g. Windows
            # without a C++ compiler). Uses confidence-weighted truncated SVD on the
            # implicit-feedback matrix as a coarse approximation of iALS. Production
            # training (Colab) installs `implicit` cleanly and uses the real path.
            log.warning("implicit lib unavailable; falling back to truncated-SVD on "
                        "confidence-weighted implicit matrix (smoke/dev only).")
            from scipy.sparse.linalg import svds
            conf = ui_matrix * float(self.alpha)
            k = min(self.n_factors, min(conf.shape) - 1)
            if k < 1:
                self.U = np.zeros((len(user_ids), 1), dtype=np.float32)
                self.V = np.zeros((len(job_ids), 1), dtype=np.float32)
            else:
                rng = np.random.default_rng(self.seed)
                v0 = rng.standard_normal(min(conf.shape)).astype(np.float32)
                U, s, Vt = svds(conf.astype(np.float32), k=k, v0=v0)
                self.U = (U * np.sqrt(s)).astype(np.float32)
                self.V = (Vt.T * np.sqrt(s)).astype(np.float32)
            log.info("CF fallback fit: %d users, %d jobs, factors=%d",
                     len(user_ids), len(job_ids), self.n_factors)
        return self

    def score_pairs(self, user_id: int, job_ids: list[int]) -> np.ndarray:
        ui = self._user_index.get(int(user_id))
        if ui is None or self.U is None or self.V is None:
            return np.zeros(len(job_ids), dtype=np.float32)
        cols = np.fromiter((self._job_index.get(int(j), -1) for j in job_ids),
                           dtype=np.int64, count=len(job_ids))
        scores = np.zeros(len(job_ids), dtype=np.float32)
        valid = cols >= 0
        scores[valid] = self.V[cols[valid]] @ self.U[ui]
        return scores

    def recommend(self, user_id: int, k: int = 10,
                  exclude: set[int] | None = None) -> list[tuple[int, float]]:
        if self._all_job_ids is None:
            return []
        ui = self._user_index.get(int(user_id))
        if ui is None or self.U is None or self.V is None:
            return []
        scores = self.V @ self.U[ui]
        if exclude:
            for jid in exclude:
                idx = self._job_index.get(int(jid))
                if idx is not None:
                    scores[idx] = -np.inf
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(self._all_job_ids[i]), float(scores[i])) for i in top]

    def knows_user(self, user_id: int) -> bool:
        return int(user_id) in self._trained_users

    # Compat shim — hybrid currently calls predict(). Returns the raw factor
    # dot product; downstream uses min-max so absolute scale doesn't matter.
    def predict(self, user_id: int, job_id: int) -> float:
        return float(self.score_pairs(user_id, [job_id])[0])

    @property
    def global_mean(self) -> float:
        # Legacy compat — implicit feedback has no meaningful "mean rating".
        return 0.0

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.savez(path / "ials.npz",
                 U=self.U, V=self.V,
                 user_ids=np.array(list(self._user_index.keys()), dtype=np.int64),
                 job_ids=np.array(self._all_job_ids, dtype=np.int64),
                 trained_users=np.array(list(self._trained_users), dtype=np.int64))

    def load(self, path: Path) -> "CollaborativeRecommender":
        # Legacy artifacts saved svd.npz; new ones save ials.npz.
        f = path / "ials.npz" if (path / "ials.npz").exists() else path / "svd.npz"
        z = np.load(f)
        self.U = np.asarray(z["U"], dtype=np.float32)
        # Older SVD save stored Vt of shape (k, n_jobs); iALS stores V (n_jobs, k).
        if "V" in z.files:
            self.V = np.asarray(z["V"], dtype=np.float32)
        else:
            self.V = np.asarray(z["Vt"], dtype=np.float32).T
        user_ids = z["user_ids"]
        self._user_index = {int(u): i for i, u in enumerate(user_ids)}
        self._all_job_ids = z["job_ids"]
        self._job_index = {int(j): i for i, j in enumerate(self._all_job_ids)}
        self._trained_users = set(int(u) for u in z["trained_users"])
        return self
