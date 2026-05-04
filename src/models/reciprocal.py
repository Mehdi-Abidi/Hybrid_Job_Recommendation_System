"""Reciprocal recommendation: model the recruiter side as well as the seeker side.

Bilateral score = sigmoid(s_user_to_job) * sigmoid(s_job_to_user) — the product of
two independent towers' agreement, where:
  - s_user_to_job is the existing two-tower forward score (user prefers job).
  - s_job_to_user is the *inverse* tower's score (recruiter for this job would
    shortlist this candidate).

The inverse direction is trained on the same positive (user, job) interaction pairs
flipped: each batch's diagonal is the positive pair under the inverse roles, and the
cross-entropy in-batch softmax learns who-prefers-whom from the recruiter's viewpoint.
We deliberately reuse `TwoTowerArtifacts` (feature matrices + id arrays) so the inverse
tower trains on identical inputs — only the loss role is flipped.

References:
  - "Revisiting Reciprocal Recommender Systems" (SIGKDD 2024)
  - MIRROR (SIGIR 2024) — bilateral preference modeling
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.two_tower import TwoTowerArtifacts, TwoTowerModel, _PairDataset
from src.utils.device import best_device
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ReciprocalConfig:
    hidden_dims: tuple[int, ...] = (256, 128)
    embedding_dim: int = 128
    dropout: float = 0.2
    epochs: int = 10
    batch_size: int = 512
    lr: float = 1e-3
    seed: int = 42


class JobToUserTower:
    """Inverse-direction two-tower trained on flipped positive pairs.

    Architecturally identical to the forward TwoTowerModel — the difference is purely
    in how the loss is computed during training (job is query, user is candidate). At
    inference we expose `score_pairs(user_id, job_ids)` returning the recruiter-side
    raw dot products (not yet sigmoid-squashed); the BilateralScorer applies sigmoid.
    """

    def __init__(self, artifacts: TwoTowerArtifacts, cfg: ReciprocalConfig | None = None):
        self.artifacts = artifacts
        self.cfg = cfg or ReciprocalConfig()
        self.model: TwoTowerModel | None = None
        self._user_id_to_row = {int(u): i for i, u in enumerate(artifacts.user_ids)}
        self._job_id_to_row = {int(j): i for i, j in enumerate(artifacts.job_ids)}

    def fit(self, train_pairs: pd.DataFrame) -> "JobToUserTower":
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        u_idx = np.fromiter(
            (self._user_id_to_row[int(u)] for u in train_pairs["user_id"]),
            dtype=np.int64, count=len(train_pairs),
        )
        j_idx = np.fromiter(
            (self._job_id_to_row[int(j)] for j in train_pairs["job_id"]),
            dtype=np.int64, count=len(train_pairs),
        )
        ds = _PairDataset(self.artifacts.user_feature_matrix, self.artifacts.job_feature_matrix, u_idx, j_idx)
        loader = DataLoader(ds, batch_size=min(cfg.batch_size, len(ds)), shuffle=True, drop_last=False)

        device = best_device()
        # Inverse roles: the model's user_tower becomes the "candidate-fit" tower
        # (input: user features), and job_tower becomes the "recruiter-preference"
        # tower (input: job features). The class is symmetric, so we keep the same
        # TwoTowerModel; only the cross-entropy direction is flipped below.
        model = TwoTowerModel(
            user_in_dim=self.artifacts.config["user_in_dim"],
            job_in_dim=self.artifacts.config["job_in_dim"],
            hidden_dims=list(cfg.hidden_dims), emb_dim=cfg.embedding_dim, dropout=cfg.dropout,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        log.info("ReciprocalTrainer (job→user) on %s, %d pairs", device, len(ds))

        model.train()
        for ep in range(cfg.epochs):
            total = 0.0
            for u_feat, j_feat in loader:
                u_feat, j_feat = u_feat.to(device), j_feat.to(device)
                u_emb = model.user_embedding(u_feat)   # candidate-fit emb
                j_emb = model.job_embedding(j_feat)    # recruiter-pref emb
                # Forward two-tower trains rows = users; here the "query" is the job.
                # Logits: (B_jobs, B_users); diagonal is positive pair under flipped roles.
                logits = j_emb @ u_emb.T
                labels = torch.arange(j_emb.size(0), device=device)
                loss = F.cross_entropy(logits, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.item()) * j_emb.size(0)
            log.info("Reciprocal epoch %d/%d loss=%.4f", ep + 1, cfg.epochs, total / max(len(ds), 1))

        self.model = model
        return self

    @torch.no_grad()
    def _embeddings(self, feat: np.ndarray, tower: str) -> np.ndarray:
        assert self.model is not None
        self.model.eval()
        device = next(self.model.parameters()).device
        x = torch.from_numpy(feat).float().to(device)
        emb = self.model.user_embedding(x) if tower == "user" else self.model.job_embedding(x)
        return emb.cpu().numpy().astype(np.float32)

    def user_embeddings(self) -> np.ndarray:
        return self._embeddings(self.artifacts.user_feature_matrix, "user")

    def job_embeddings(self) -> np.ndarray:
        return self._embeddings(self.artifacts.job_feature_matrix, "job")

    # Recruiter-side dot products for one user against many jobs. Returns raw scores
    # (pre-sigmoid). Unknown ids → 0.0, mirroring the forward tower's policy.
    def score_pairs(self, user_id: int, job_ids: list[int]) -> np.ndarray:
        if self.model is None:
            return np.zeros(len(job_ids), dtype=np.float32)
        u_row = self._user_id_to_row.get(int(user_id), -1)
        if u_row < 0:
            return np.zeros(len(job_ids), dtype=np.float32)
        u_emb = self.user_embeddings()[u_row]
        j_emb = self.job_embeddings()
        out = np.zeros(len(job_ids), dtype=np.float32)
        for i, jid in enumerate(job_ids):
            j_row = self._job_id_to_row.get(int(jid), -1)
            if j_row >= 0:
                out[i] = float(j_emb[j_row] @ u_emb)
        return out

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            torch.save(self.model.state_dict(), path / "reciprocal.pt")

    def load(self, path: Path) -> "JobToUserTower":
        device = best_device()
        cfg = self.cfg
        model = TwoTowerModel(
            user_in_dim=self.artifacts.config["user_in_dim"],
            job_in_dim=self.artifacts.config["job_in_dim"],
            hidden_dims=list(cfg.hidden_dims), emb_dim=cfg.embedding_dim, dropout=cfg.dropout,
        ).to(device)
        model.load_state_dict(torch.load(path / "reciprocal.pt", map_location=device))
        self.model = model
        return self


def _sigmoid(x: np.ndarray | float) -> np.ndarray:
    # Stable sigmoid for both scalars and arrays.
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class BilateralScorer:
    """Combines forward (user→job) and inverse (job→user) scores into a bilateral score.

    Scoring policy: each direction's raw dot product is squashed with sigmoid into a
    pseudo-probability, and the bilateral score is their product. Multiplication
    (rather than mean) penalises one-sided fits — a job a user wants but the recruiter
    wouldn't shortlist gets a low bilateral score, even if user-side affinity is high.
    """

    def __init__(self, forward_tower, inverse_tower: JobToUserTower):
        # `forward_tower` is a TwoTowerTrainer (already trained); we access its .model
        # and feature matrices through .artifacts to score pairs without re-encoding.
        self.fwd = forward_tower
        self.inv = inverse_tower

    def _forward_scores(self, user_id: int, job_ids: list[int]) -> np.ndarray:
        art = self.fwd.artifacts
        u_pos = np.where(art.user_ids == int(user_id))[0]
        if len(u_pos) == 0:
            return np.zeros(len(job_ids), dtype=np.float32)
        ue = self.fwd.user_embeddings()[u_pos[0]]
        je = self.fwd.job_embeddings()
        j_id_to_row = {int(j): i for i, j in enumerate(art.job_ids)}
        out = np.zeros(len(job_ids), dtype=np.float32)
        for i, jid in enumerate(job_ids):
            row = j_id_to_row.get(int(jid), -1)
            if row >= 0:
                out[i] = float(je[row] @ ue)
        return out

    # Returns (s_user_to_job, s_job_to_user, bilateral) all as np.ndarray of shape (n,).
    # Raw forward/inverse scores are returned unsquashed so the LTR feature builder can
    # use them directly; bilateral is sigmoid(fwd) * sigmoid(inv).
    def score(self, user_id: int, job_ids: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not job_ids:
            empty = np.zeros(0, dtype=np.float32)
            return empty, empty, empty
        fwd = self._forward_scores(int(user_id), job_ids)
        inv = self.inv.score_pairs(int(user_id), job_ids)
        bilateral = (_sigmoid(fwd) * _sigmoid(inv)).astype(np.float32)
        return fwd, inv, bilateral
