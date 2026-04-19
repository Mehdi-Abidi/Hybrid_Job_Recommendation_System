"""Mult-VAE: variational autoencoder with multinomial likelihood for collaborative filtering.
Each user is represented by their binary interaction vector over items; the VAE learns a
low-dim latent that regenerates the user's full preference distribution."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class MultVAEConfig:
    hidden_dim: int = 256
    latent_dim: int = 64
    dropout: float = 0.5
    batch_size: int = 128
    epochs: int = 20
    lr: float = 1e-3
    anneal_steps: int = 2000
    max_beta: float = 0.2
    seed: int = 42


class MultVAE(nn.Module):
    def __init__(self, n_items: int, cfg: MultVAEConfig):
        super().__init__()
        self.n_items = n_items
        self.encoder = nn.Sequential(
            nn.Linear(n_items, cfg.hidden_dim), nn.Tanh(), nn.Dropout(cfg.dropout),
        )
        self.mu_head = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.logvar_head = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim), nn.Tanh(),
            nn.Linear(cfg.hidden_dim, n_items),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # L2-normalize inputs as in the original paper.
        x_norm = F.normalize(x, p=2, dim=1)
        h = self.encoder(x_norm)
        mu = self.mu_head(h); logvar = self.logvar_head(h)
        if self.training:
            std = (0.5 * logvar).exp()
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        logits = self.decoder(z)
        return logits, mu, logvar


class MultVAERecommender:
    def __init__(self, cfg: MultVAEConfig):
        self.cfg = cfg
        self.model: MultVAE | None = None
        self._user_idx: dict[int, int] = {}
        self._job_idx: dict[int, int] = {}
        self._user_inputs: np.ndarray | None = None
        self._jobs: pd.DataFrame | None = None

    def fit(self, users: pd.DataFrame, jobs: pd.DataFrame, train: pd.DataFrame) -> "MultVAERecommender":
        torch.manual_seed(self.cfg.seed)
        self._user_idx = {int(u): i for i, u in enumerate(users["user_id"])}
        self._job_idx = {int(j): i for i, j in enumerate(jobs["job_id"])}
        self._jobs = jobs
        n_users, n_items = len(self._user_idx), len(self._job_idx)
        X = np.zeros((n_users, n_items), dtype=np.float32)
        for _, r in train.iterrows():
            u = self._user_idx.get(int(r["user_id"])); j = self._job_idx.get(int(r["job_id"]))
            if u is not None and j is not None:
                X[u, j] = float(r["rating"])
        self._user_inputs = X

        from src.utils.device import best_device
        device = best_device()
        self.model = MultVAE(n_items, self.cfg).to(device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        X_t = torch.from_numpy(X).to(device)
        log.info("MultVAE training on %s", device)

        step = 0
        self.model.train()
        for ep in range(self.cfg.epochs):
            idx = torch.randperm(n_users)
            total = 0.0
            for s in range(0, n_users, self.cfg.batch_size):
                batch = idx[s:s + self.cfg.batch_size]
                xb = X_t[batch]
                logits, mu, logvar = self.model(xb)
                log_probs = F.log_softmax(logits, dim=1)
                # Multinomial log-likelihood (normalize targets so rows sum to 1).
                target = xb / (xb.sum(dim=1, keepdim=True) + 1e-12)
                recon = -(target * log_probs).sum(dim=1).mean()
                kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
                beta = min(self.cfg.max_beta, self.cfg.max_beta * step / max(self.cfg.anneal_steps, 1))
                loss = recon + beta * kl
                opt.zero_grad(); loss.backward(); opt.step()
                step += 1
                total += float(loss.item()) * xb.size(0)
            log.info("MultVAE ep %d/%d loss=%.4f", ep + 1, self.cfg.epochs, total / max(n_users, 1))
        return self

    @torch.no_grad()
    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = True) -> list[tuple[int, float]]:
        u = self._user_idx.get(int(user_id))
        if u is None or self.model is None or self._user_inputs is None:
            return []
        self.model.eval()
        device = next(self.model.parameters()).device
        x = torch.from_numpy(self._user_inputs[u:u + 1]).to(device)
        logits, _, _ = self.model(x)
        scores = logits[0].cpu().numpy()
        if exclude_seen:
            scores[self._user_inputs[u] > 0] = -np.inf
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        inv = {i: j for j, i in self._job_idx.items()}
        return [(int(inv[i]), float(scores[i])) for i in top]

    @torch.no_grad()
    def score_pairs(self, user_id: int, job_ids: list[int]) -> np.ndarray:
        u = self._user_idx.get(int(user_id))
        if u is None or self.model is None:
            return np.zeros(len(job_ids), dtype=np.float32)
        self.model.eval()
        device = next(self.model.parameters()).device
        x = torch.from_numpy(self._user_inputs[u:u + 1]).to(device)
        logits, _, _ = self.model(x)
        scores = logits[0].cpu().numpy()
        return np.array([scores[self._job_idx[int(j)]] if int(j) in self._job_idx else 0.0
                         for j in job_ids], dtype=np.float32)
