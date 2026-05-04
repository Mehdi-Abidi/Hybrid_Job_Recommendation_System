"""Two-Tower neural retrieval: user and job towers share an embedding space.
Training uses in-batch sampled softmax over positive (user, job) interaction pairs."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.features.structured_features import (
    SkillEncoder, CategoricalEncoder, HashBucketEncoder, bin_experience, parse_location,
)
from src.features.text_features import EmbeddingFeaturizer, job_text, user_text
from src.utils.device import best_device
from src.utils.logging import get_logger

log = get_logger(__name__)


# Tower MLP: [text_emb || structured_features] → emb_dim, L2-normalized.
class _Tower(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], out_dim: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, user_in_dim: int, job_in_dim: int, hidden_dims: list[int],
                 emb_dim: int, dropout: float = 0.2):
        super().__init__()
        self.user_tower = _Tower(user_in_dim, hidden_dims, emb_dim, dropout)
        self.job_tower = _Tower(job_in_dim, hidden_dims, emb_dim, dropout)
        self.emb_dim = emb_dim

    def user_embedding(self, user_features: torch.Tensor) -> torch.Tensor:
        return self.user_tower(user_features)

    def job_embedding(self, job_features: torch.Tensor) -> torch.Tensor:
        return self.job_tower(job_features)

    def forward(self, user_features: torch.Tensor, job_features: torch.Tensor) -> torch.Tensor:
        u = self.user_embedding(user_features)
        j = self.job_embedding(job_features)
        return (u * j).sum(dim=-1)


@dataclass
class TwoTowerArtifacts:
    skill_enc: SkillEncoder
    cat_enc: CategoricalEncoder            # job category (small vocab — keep as one-hot)
    loc_hash: HashBucketEncoder            # location → 256 hash buckets (replaces 10k+ one-hot)
    state_enc: CategoricalEncoder          # parsed state, small vocab
    country_enc: CategoricalEncoder        # parsed country, small vocab
    user_text_emb_dim: int
    job_text_emb_dim: int
    user_feature_matrix: np.ndarray        # (n_users, user_in_dim) float32
    job_feature_matrix: np.ndarray         # (n_jobs, job_in_dim) float32
    user_ids: np.ndarray
    job_ids: np.ndarray
    config: dict[str, Any] = field(default_factory=dict)


# Pairs dataset: each sample is (user_row_idx, job_row_idx) for a positive interaction.
class _PairDataset(Dataset):
    def __init__(self, user_feat: np.ndarray, job_feat: np.ndarray,
                 user_idx: np.ndarray, job_idx: np.ndarray):
        self.user_feat = torch.from_numpy(user_feat).float()
        self.job_feat = torch.from_numpy(job_feat).float()
        self.user_idx = torch.from_numpy(user_idx).long()
        self.job_idx = torch.from_numpy(job_idx).long()

    def __len__(self): return len(self.user_idx)

    def __getitem__(self, i):
        return self.user_feat[self.user_idx[i]], self.job_feat[self.job_idx[i]]


class TwoTowerTrainer:
    """End-to-end: build features, train with in-batch softmax, expose embeddings."""

    def __init__(self, cfg: dict[str, Any], text_embedder: EmbeddingFeaturizer | None = None, seed: int = 42):
        self.cfg = cfg
        self.text_embedder = text_embedder or EmbeddingFeaturizer()
        self.seed = seed
        self.model: TwoTowerModel | None = None
        self.artifacts: TwoTowerArtifacts | None = None

    # Build structured+text feature matrices for all jobs and users.
    # Location features: 256-bucket hash one-hot + state + country (parsed from
    # "City, State, Country"). Cuts the 10k+ raw-location one-hot to ~256+small,
    # avoids the ~7GB feature-matrix blowup, and gives geographic generalization
    # (Facebook KDD 2020 hash-bucket recipe).
    def build_features(self, jobs: pd.DataFrame, users: pd.DataFrame) -> TwoTowerArtifacts:
        N_LOC_BUCKETS = 256
        skill_enc = SkillEncoder().fit(list(jobs["skills"].astype(str)) + list(users["skills"].astype(str)))
        cat_enc = (CategoricalEncoder().fit(list(jobs["category"].astype(str)))
                   if "category" in jobs.columns else CategoricalEncoder().fit([""]))
        loc_hash = HashBucketEncoder(N_LOC_BUCKETS)
        all_locs = list(jobs["location"].astype(str))
        if "preferred_location" in users.columns:
            all_locs += list(users["preferred_location"].astype(str))
        state_enc = CategoricalEncoder().fit([parse_location(l)[1] for l in all_locs])
        country_enc = CategoricalEncoder().fit([parse_location(l)[2] for l in all_locs])

        def _loc_features(loc_strings: list[str]) -> np.ndarray:
            buckets = loc_hash.transform(loc_strings)
            bucket_oh = np.eye(N_LOC_BUCKETS, dtype=np.float32)[buckets]
            states = [parse_location(l)[1] for l in loc_strings]
            countries = [parse_location(l)[2] for l in loc_strings]
            state_oh = np.eye(state_enc.size, dtype=np.float32)[state_enc.transform(states)]
            country_oh = np.eye(country_enc.size, dtype=np.float32)[country_enc.transform(countries)]
            return np.hstack([bucket_oh, state_oh, country_oh]).astype(np.float32)

        job_skill = skill_enc.transform(list(jobs["skills"].astype(str)))
        job_cat = (cat_enc.transform(list(jobs["category"].astype(str)))
                   if "category" in jobs.columns else np.zeros(len(jobs), dtype=np.int64))
        job_cat_1h = np.eye(cat_enc.size, dtype=np.float32)[job_cat]
        job_loc_feat = _loc_features(list(jobs["location"].astype(str)))
        job_text_emb = self.text_embedder.encode([job_text(r) for _, r in jobs.iterrows()], normalize=True)
        job_feat = np.hstack([job_skill, job_cat_1h, job_loc_feat, job_text_emb]).astype(np.float32)

        user_skill = skill_enc.transform(list(users["skills"].astype(str)))
        user_exp = bin_experience(users["experience_years"]).reshape(-1, 1).astype(np.float32)
        if "preferred_location" in users.columns:
            user_loc_feat = _loc_features(list(users["preferred_location"].astype(str)))
        else:
            user_loc_feat = np.zeros(
                (len(users), N_LOC_BUCKETS + state_enc.size + country_enc.size),
                dtype=np.float32,
            )
        user_text_emb = self.text_embedder.encode([user_text(r) for _, r in users.iterrows()], normalize=True)
        user_feat = np.hstack([user_skill, user_exp, user_loc_feat, user_text_emb]).astype(np.float32)

        self.artifacts = TwoTowerArtifacts(
            skill_enc=skill_enc, cat_enc=cat_enc, loc_hash=loc_hash,
            state_enc=state_enc, country_enc=country_enc,
            user_text_emb_dim=user_text_emb.shape[1], job_text_emb_dim=job_text_emb.shape[1],
            user_feature_matrix=user_feat, job_feature_matrix=job_feat,
            user_ids=users["user_id"].to_numpy(), job_ids=jobs["job_id"].to_numpy(),
            config={"user_in_dim": user_feat.shape[1], "job_in_dim": job_feat.shape[1]},
        )
        return self.artifacts

    # Train two-tower with in-batch softmax over positive pairs.
    def train(self, train_pairs: pd.DataFrame, epochs: int | None = None,
              batch_size: int | None = None, lr: float | None = None) -> TwoTowerModel:
        assert self.artifacts is not None, "Call build_features first."
        cfg = self.cfg
        epochs = epochs or cfg.get("epochs", 10)
        batch_size = batch_size or cfg.get("batch_size", 512)
        lr = lr or cfg.get("lr", 1e-3)
        hidden_dims = cfg.get("hidden_dims", [256, 128])
        emb_dim = cfg.get("embedding_dim", 128)
        dropout = cfg.get("dropout", 0.2)

        torch.manual_seed(self.seed)
        u_id_to_row = {int(u): i for i, u in enumerate(self.artifacts.user_ids)}
        j_id_to_row = {int(j): i for i, j in enumerate(self.artifacts.job_ids)}
        u_idx = np.fromiter((u_id_to_row[int(u)] for u in train_pairs["user_id"]),
                            dtype=np.int64, count=len(train_pairs))
        j_idx = np.fromiter((j_id_to_row[int(j)] for j in train_pairs["job_id"]),
                            dtype=np.int64, count=len(train_pairs))
        ds = _PairDataset(self.artifacts.user_feature_matrix, self.artifacts.job_feature_matrix, u_idx, j_idx)
        loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True, drop_last=False)

        device = best_device()
        model = TwoTowerModel(
            user_in_dim=self.artifacts.config["user_in_dim"],
            job_in_dim=self.artifacts.config["job_in_dim"],
            hidden_dims=hidden_dims, emb_dim=emb_dim, dropout=dropout,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        log.info("TwoTower training on %s", device)

        model.train()
        for ep in range(epochs):
            total = 0.0
            for u_feat, j_feat in loader:
                u_feat = u_feat.to(device); j_feat = j_feat.to(device)
                u = model.user_embedding(u_feat)
                j = model.job_embedding(j_feat)
                logits = u @ j.T  # (B, B); positives on the diagonal
                labels = torch.arange(u.size(0), device=device)
                loss = F.cross_entropy(logits, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.item()) * u.size(0)
            log.info("TwoTower epoch %d/%d loss=%.4f", ep + 1, epochs, total / max(len(ds), 1))

        self.model = model
        return model

    # All job embeddings (n_jobs, emb_dim), L2-normalized, for FAISS indexing.
    @torch.no_grad()
    def job_embeddings(self) -> np.ndarray:
        assert self.model is not None and self.artifacts is not None
        self.model.eval()
        device = next(self.model.parameters()).device
        j = torch.from_numpy(self.artifacts.job_feature_matrix).float().to(device)
        return self.model.job_embedding(j).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def user_embeddings(self) -> np.ndarray:
        assert self.model is not None and self.artifacts is not None
        self.model.eval()
        device = next(self.model.parameters()).device
        u = torch.from_numpy(self.artifacts.user_feature_matrix).float().to(device)
        return self.model.user_embedding(u).cpu().numpy().astype(np.float32)

    def save(self, path: Path) -> None:
        import joblib
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "two_tower.pt")
        joblib.dump({
            "skill_enc": self.artifacts.skill_enc, "cat_enc": self.artifacts.cat_enc,
            "loc_hash": self.artifacts.loc_hash, "state_enc": self.artifacts.state_enc,
            "country_enc": self.artifacts.country_enc, "config": self.artifacts.config,
            "user_ids": self.artifacts.user_ids, "job_ids": self.artifacts.job_ids,
        }, path / "two_tower_artifacts.joblib")
        np.save(path / "two_tower_user_feat.npy", self.artifacts.user_feature_matrix)
        np.save(path / "two_tower_job_feat.npy", self.artifacts.job_feature_matrix)

    def load(self, path: Path) -> "TwoTowerTrainer":
        import joblib
        obj = joblib.load(path / "two_tower_artifacts.joblib")
        user_feat = np.load(path / "two_tower_user_feat.npy")
        job_feat = np.load(path / "two_tower_job_feat.npy")
        self.artifacts = TwoTowerArtifacts(
            skill_enc=obj["skill_enc"], cat_enc=obj["cat_enc"],
            loc_hash=obj["loc_hash"], state_enc=obj["state_enc"],
            country_enc=obj["country_enc"],
            user_text_emb_dim=0, job_text_emb_dim=0,
            user_feature_matrix=user_feat, job_feature_matrix=job_feat,
            user_ids=obj["user_ids"], job_ids=obj["job_ids"], config=obj["config"],
        )
        device = best_device()
        self.model = TwoTowerModel(
            user_in_dim=obj["config"]["user_in_dim"], job_in_dim=obj["config"]["job_in_dim"],
            hidden_dims=self.cfg.get("hidden_dims", [256, 128]),
            emb_dim=self.cfg.get("embedding_dim", 128), dropout=self.cfg.get("dropout", 0.2),
        ).to(device)
        self.model.load_state_dict(torch.load(path / "two_tower.pt", map_location=device))
        return self
