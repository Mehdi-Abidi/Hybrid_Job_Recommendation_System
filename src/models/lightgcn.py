"""LightGCN: collaborative filtering on the user-item bipartite graph.
Each layer: e^(k+1) = Â * e^(k). Final embedding = mean across layers. No self-loops, no non-linearity."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix

from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class LightGCNConfig:
    emb_dim: int = 64
    n_layers: int = 3
    batch_size: int = 1024
    epochs: int = 20
    lr: float = 1e-3
    reg: float = 1e-4
    seed: int = 42


def _build_norm_adj(user_job: coo_matrix, n_users: int, n_jobs: int) -> torch.Tensor:
    # A_hat = D^-1/2 A D^-1/2 over the bipartite graph with no self loops.
    rows = np.concatenate([user_job.row, user_job.col + n_users])
    cols = np.concatenate([user_job.col + n_users, user_job.row])
    data = np.ones(len(rows), dtype=np.float32)
    n = n_users + n_jobs
    deg = np.bincount(rows, minlength=n).astype(np.float32)
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(np.where(deg > 0, deg, 1.0)), 0.0)
    norm = d_inv_sqrt[rows] * d_inv_sqrt[cols]
    indices = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
    values = torch.tensor(norm * data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()


class LightGCN(nn.Module):
    def __init__(self, n_users: int, n_jobs: int, cfg: LightGCNConfig, adj: torch.Tensor):
        super().__init__()
        self.n_users = n_users
        self.n_jobs = n_jobs
        self.cfg = cfg
        self.emb = nn.Embedding(n_users + n_jobs, cfg.emb_dim)
        nn.init.xavier_uniform_(self.emb.weight)
        self.register_buffer("adj", adj, persistent=False)

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        e = self.emb.weight
        layers = [e]
        for _ in range(self.cfg.n_layers):
            e = torch.sparse.mm(self.adj, e)
            layers.append(e)
        final = torch.stack(layers, 0).mean(0)
        return final[: self.n_users], final[self.n_users:]

    def bpr_loss(self, u: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        ue, je = self.propagate()
        uv = ue[u]; pv = je[pos]; nv = je[neg]
        pos_score = (uv * pv).sum(-1)
        neg_score = (uv * nv).sum(-1)
        loss = -F.logsigmoid(pos_score - neg_score).mean()
        reg = self.cfg.reg * (uv.pow(2).sum() + pv.pow(2).sum() + nv.pow(2).sum()) / uv.size(0)
        return loss + reg


class LightGCNRecommender:
    def __init__(self, cfg: LightGCNConfig):
        self.cfg = cfg
        self.model: LightGCN | None = None
        self._user_idx: dict[int, int] = {}
        self._job_idx: dict[int, int] = {}
        self._user_pos: dict[int, set[int]] = {}
        self._all_job_rows: np.ndarray | None = None

    def fit(self, users: pd.DataFrame, jobs: pd.DataFrame, train: pd.DataFrame) -> "LightGCNRecommender":
        torch.manual_seed(self.cfg.seed)
        rng = np.random.default_rng(self.cfg.seed)

        self._user_idx = {int(u): i for i, u in enumerate(users["user_id"])}
        self._job_idx = {int(j): i for i, j in enumerate(jobs["job_id"])}
        n_users, n_jobs = len(self._user_idx), len(self._job_idx)
        self._all_job_rows = np.arange(n_jobs)

        rows = np.array([self._user_idx[int(r["user_id"])] for _, r in train.iterrows()
                         if int(r["user_id"]) in self._user_idx and int(r["job_id"]) in self._job_idx])
        cols = np.array([self._job_idx[int(r["job_id"])] for _, r in train.iterrows()
                         if int(r["user_id"]) in self._user_idx and int(r["job_id"]) in self._job_idx])
        if len(rows) == 0:
            log.warning("LightGCN: no training edges."); return self
        mat = coo_matrix((np.ones_like(rows, dtype=np.float32), (rows, cols)), shape=(n_users, n_jobs))
        self._user_pos = {u: set() for u in range(n_users)}
        for u, j in zip(rows, cols):
            self._user_pos[int(u)].add(int(j))

        from src.utils.device import best_device
        device = best_device()
        adj = _build_norm_adj(mat, n_users, n_jobs).to(device)
        self.model = LightGCN(n_users, n_jobs, self.cfg, adj).to(device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        log.info("LightGCN training on %s", device)

        # BPR training with uniform random negatives.
        user_pool = rows
        pos_pool = cols

        self.model.train()
        for ep in range(self.cfg.epochs):
            perm = rng.permutation(len(user_pool))
            total, n = 0.0, 0
            for start in range(0, len(perm), self.cfg.batch_size):
                batch = perm[start:start + self.cfg.batch_size]
                ub = torch.tensor(user_pool[batch], dtype=torch.long, device=device)
                pb = torch.tensor(pos_pool[batch], dtype=torch.long, device=device)
                neg = np.empty_like(batch)
                for i, u in enumerate(user_pool[batch]):
                    while True:
                        jn = int(rng.integers(0, n_jobs))
                        if jn not in self._user_pos[int(u)]:
                            neg[i] = jn; break
                nb = torch.tensor(neg, dtype=torch.long, device=device)
                loss = self.model.bpr_loss(ub, pb, nb)
                opt.zero_grad(); loss.backward(); opt.step()
                total += float(loss.item()) * len(batch); n += len(batch)
            log.info("LightGCN ep %d/%d loss=%.4f", ep + 1, self.cfg.epochs, total / max(n, 1))
        return self

    @torch.no_grad()
    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = True) -> list[tuple[int, float]]:
        u = self._user_idx.get(int(user_id))
        if u is None or self.model is None:
            return []
        self.model.eval()
        ue, je = self.model.propagate()
        scores = (je @ ue[u]).cpu().numpy()
        if exclude_seen:
            for j in self._user_pos.get(u, set()):
                scores[j] = -np.inf
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
        ue, je = self.model.propagate()
        out = np.zeros(len(job_ids), dtype=np.float32)
        for i, jid in enumerate(job_ids):
            j = self._job_idx.get(int(jid))
            if j is not None:
                out[i] = float(ue[u] @ je[j])
        return out
