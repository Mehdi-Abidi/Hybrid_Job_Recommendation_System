"""DeepFM: factorization machine + deep MLP over sparse structured features.
For job recommendation we use (user_id, job_id, user_category, job_category, user_location,
job_location, skill overlap bin, experience bin) as the categorical field inputs."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.features.structured_features import parse_skills
from src.utils.device import best_device
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class DeepFMConfig:
    emb_dim: int = 8
    mlp_dims: tuple[int, ...] = (64, 32)
    dropout: float = 0.1
    batch_size: int = 256
    epochs: int = 10
    lr: float = 1e-3
    n_negatives: int = 4
    seed: int = 42


class DeepFM(nn.Module):
    def __init__(self, field_dims: list[int], emb_dim: int, mlp_dims: tuple[int, ...], dropout: float):
        super().__init__()
        self.field_dims = field_dims
        self.n_fields = len(field_dims)
        # Offsets let one embedding table cover all fields.
        offsets = np.cumsum([0] + field_dims[:-1])
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        self.embedding = nn.Embedding(sum(field_dims), emb_dim)
        self.linear = nn.Embedding(sum(field_dims), 1)
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.zeros_(self.linear.weight)

        mlp_input = self.n_fields * emb_dim
        layers: list[nn.Module] = []
        for h in mlp_dims:
            layers += [nn.Linear(mlp_input, h), nn.ReLU(), nn.Dropout(dropout)]
            mlp_input = h
        layers.append(nn.Linear(mlp_input, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_fields) long.
        x = x + self.offsets
        e = self.embedding(x)                # (B, F, D)
        linear = self.linear(x).sum(1).squeeze(-1)
        # FM second-order: 0.5 * (sum_e^2 - sum_e_squared).
        sq_sum = e.sum(1).pow(2).sum(-1)
        sum_sq = e.pow(2).sum(1).sum(-1)
        fm = 0.5 * (sq_sum - sum_sq)
        deep = self.mlp(e.view(e.size(0), -1)).squeeze(-1)
        return linear + fm + deep


class DeepFMRecommender:
    """Sampled-softmax style: positives from train interactions, uniform random negatives per positive."""

    def __init__(self, cfg: DeepFMConfig):
        self.cfg = cfg
        self.model: DeepFM | None = None
        self._user_idx: dict[int, int] = {}
        self._job_idx: dict[int, int] = {}
        self._field_dims: list[int] = []
        # Pre-computed field values per user and per job.
        self._user_fields: np.ndarray | None = None    # (n_users, 3) — user_id, category, location
        self._job_fields: np.ndarray | None = None     # (n_jobs, 3)  — job_id, category, location
        self._user_skills: list[set[str]] = []
        self._job_skills: list[set[str]] = []
        self._user_exp_bin: np.ndarray | None = None
        self._jobs: pd.DataFrame | None = None

    # Build a fixed field vocabulary and precompute per-user/per-job field indices.
    def _build_fields(self, users: pd.DataFrame, jobs: pd.DataFrame):
        self._user_idx = {int(u): i for i, u in enumerate(users["user_id"])}
        self._job_idx = {int(j): i for i, j in enumerate(jobs["job_id"])}

        cats = sorted(set(map(str, jobs.get("category", pd.Series([""])))) |
                      set(map(str, users.get("primary_category", pd.Series([""])))))
        locs = sorted(set(map(str, jobs.get("location", pd.Series([""])))) |
                      set(map(str, users.get("preferred_location", pd.Series([""])))))
        cat_idx = {c: i for i, c in enumerate(cats)}
        loc_idx = {l: i for i, l in enumerate(locs)}
        exp_bins = 5  # see structured_features.bin_experience
        skill_bins = 4  # overlap bucket: 0, 1, 2, 3+

        self._field_dims = [
            len(users), len(jobs), len(cats), len(cats), len(locs), len(locs), skill_bins, exp_bins,
        ]
        self._cat_idx, self._loc_idx = cat_idx, loc_idx
        self._skill_bins = skill_bins
        self._exp_bins = exp_bins

        # Pre-compute static user/job fields (non-cross).
        self._user_fields = np.array([[
            self._user_idx[int(r["user_id"])],
            cat_idx.get(str(r.get("primary_category", "")), 0),
            loc_idx.get(str(r.get("preferred_location", "")), 0),
        ] for _, r in users.iterrows()], dtype=np.int64)

        self._job_fields = np.array([[
            self._job_idx[int(r["job_id"])],
            cat_idx.get(str(r.get("category", "")), 0),
            loc_idx.get(str(r.get("location", "")), 0),
        ] for _, r in jobs.iterrows()], dtype=np.int64)

        self._user_skills = [parse_skills(str(r.get("skills", ""))) for _, r in users.iterrows()]
        self._job_skills = [parse_skills(str(r.get("skills", ""))) for _, r in jobs.iterrows()]
        years = users.get("experience_years", pd.Series([0] * len(users))).fillna(0).astype(int).clip(0, 40)
        self._user_exp_bin = np.clip(years.to_numpy() // 5, 0, exp_bins - 1).astype(np.int64)
        self._jobs = jobs

    def _encode_pair(self, uid_row: int, jid_row: int) -> list[int]:
        u = self._user_fields[uid_row]
        j = self._job_fields[jid_row]
        overlap = len(self._user_skills[uid_row] & self._job_skills[jid_row])
        skill_bucket = min(overlap, self._skill_bins - 1)
        return [u[0], j[0], u[1], j[1], u[2], j[2], skill_bucket, int(self._user_exp_bin[uid_row])]

    def fit(self, users: pd.DataFrame, jobs: pd.DataFrame, train: pd.DataFrame) -> "DeepFMRecommender":
        self._build_fields(users, jobs)
        torch.manual_seed(self.cfg.seed)
        rng = np.random.default_rng(self.cfg.seed)

        pos_pairs = [(self._user_idx[int(r["user_id"])], self._job_idx[int(r["job_id"])])
                     for _, r in train.iterrows()
                     if int(r["user_id"]) in self._user_idx and int(r["job_id"]) in self._job_idx]
        seen = {(u, j) for u, j in pos_pairs}
        n_jobs = len(self._job_idx)

        X_rows, y_rows = [], []
        for u, j in pos_pairs:
            X_rows.append(self._encode_pair(u, j)); y_rows.append(1)
            for _ in range(self.cfg.n_negatives):
                for _attempt in range(10):
                    jn = int(rng.integers(0, n_jobs))
                    if (u, jn) not in seen:
                        X_rows.append(self._encode_pair(u, jn)); y_rows.append(0)
                        break

        X = torch.tensor(np.asarray(X_rows, dtype=np.int64))
        y = torch.tensor(np.asarray(y_rows, dtype=np.float32))
        ds = TensorDataset(X, y)
        loader = DataLoader(ds, batch_size=min(self.cfg.batch_size, len(ds)), shuffle=True)

        device = best_device()
        self.model = DeepFM(self._field_dims, self.cfg.emb_dim, self.cfg.mlp_dims, self.cfg.dropout).to(device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        log.info("DeepFM training on %s", device)

        self.model.train()
        for ep in range(self.cfg.epochs):
            total, n = 0.0, 0
            for xb, yb in loader:
                xb = xb.to(device); yb = yb.to(device)
                logit = self.model(xb)
                loss = F.binary_cross_entropy_with_logits(logit, yb)
                opt.zero_grad(); loss.backward(); opt.step()
                total += float(loss.item()) * xb.size(0); n += xb.size(0)
            log.info("DeepFM ep %d/%d loss=%.4f", ep + 1, self.cfg.epochs, total / max(n, 1))
        return self

    @torch.no_grad()
    def score_pairs(self, user_id: int, job_ids: list[int]) -> np.ndarray:
        u = self._user_idx.get(int(user_id))
        if u is None or self.model is None:
            return np.zeros(len(job_ids), dtype=np.float32)
        self.model.eval()
        rows = np.array([self._encode_pair(u, self._job_idx[int(j)]) for j in job_ids if int(j) in self._job_idx])
        if len(rows) == 0:
            return np.zeros(len(job_ids), dtype=np.float32)
        device = next(self.model.parameters()).device
        logits = self.model(torch.tensor(rows, dtype=torch.long, device=device)).cpu().numpy()
        return 1.0 / (1.0 + np.exp(-logits))

    def recommend(self, user_id: int, k: int = 10, exclude: set[int] | None = None) -> list[tuple[int, float]]:
        if self._jobs is None:
            return []
        job_ids = list(self._jobs["job_id"])
        scores = self.score_pairs(user_id, job_ids)
        exclude = exclude or set()
        if exclude:
            for i, j in enumerate(job_ids):
                if int(j) in exclude:
                    scores[i] = -np.inf
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(job_ids[i]), float(scores[i])) for i in top]
