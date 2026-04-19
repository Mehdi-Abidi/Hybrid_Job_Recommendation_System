"""BERT4Rec: bidirectional transformer for sequential recommendation.
Trains by masking random items in each user's interaction sequence and predicting them."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.utils.device import best_device
from src.utils.logging import get_logger

log = get_logger(__name__)

PAD_ID = 0
MASK_ID = 1  # reserved; real items start at index 2


@dataclass
class BERT4RecConfig:
    max_len: int = 50
    hidden_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    mask_prob: float = 0.15
    batch_size: int = 128
    epochs: int = 20
    lr: float = 1e-3
    seed: int = 42


class _SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.ln2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(x, x, x, key_padding_mask=pad_mask, need_weights=False)
        x = self.ln1(x + self.drop(a))
        x = self.ln2(x + self.drop(self.ff(x)))
        return x


class BERT4Rec(nn.Module):
    def __init__(self, n_items: int, cfg: BERT4RecConfig):
        super().__init__()
        self.cfg = cfg
        self.n_items = n_items
        vocab = n_items + 2  # +pad +mask
        self.item_emb = nn.Embedding(vocab, cfg.hidden_dim, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.hidden_dim)
        self.blocks = nn.ModuleList([
            _SelfAttentionBlock(cfg.hidden_dim, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers)
        ])
        self.drop = nn.Dropout(cfg.dropout)
        self.out = nn.Linear(cfg.hidden_dim, vocab)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: (B, L) with token ids. Returns logits (B, L, vocab).
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.drop(self.item_emb(seq) + self.pos_emb(pos))
        pad_mask = seq.eq(PAD_ID)
        for blk in self.blocks:
            x = blk(x, pad_mask)
        return self.out(x)


class _MaskedSeqDataset(Dataset):
    def __init__(self, sequences: list[list[int]], cfg: BERT4RecConfig, rng: np.random.Generator):
        self.sequences = sequences
        self.cfg = cfg
        self.rng = rng

    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx):
        s = self.sequences[idx][-self.cfg.max_len:]
        seq = [PAD_ID] * (self.cfg.max_len - len(s)) + s
        labels = [PAD_ID] * self.cfg.max_len
        for i in range(self.cfg.max_len):
            if seq[i] == PAD_ID:
                continue
            if self.rng.random() < self.cfg.mask_prob:
                labels[i] = seq[i]
                seq[i] = MASK_ID
        return torch.tensor(seq, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class BERT4RecTrainer:
    def __init__(self, cfg: BERT4RecConfig):
        self.cfg = cfg
        self.model: BERT4Rec | None = None
        self._item_to_token: dict[int, int] = {}
        self._token_to_item: dict[int, int] = {}

    # Build per-user interaction sequences ordered by recency (oldest first).
    def _build_sequences(self, interactions: pd.DataFrame) -> list[list[int]]:
        items = sorted(interactions["job_id"].unique().tolist())
        # Reserve token ids 0 (pad), 1 (mask); items start at 2.
        self._item_to_token = {int(j): i + 2 for i, j in enumerate(items)}
        self._token_to_item = {i + 2: int(j) for i, j in enumerate(items)}
        sort_col = "timestamp_days_ago" if "timestamp_days_ago" in interactions.columns else None
        out = []
        for _, g in interactions.groupby("user_id"):
            if sort_col is not None:
                g = g.sort_values(sort_col, ascending=False)  # oldest first
            out.append([self._item_to_token[int(j)] for j in g["job_id"]])
        return out

    def fit(self, train: pd.DataFrame) -> "BERT4RecTrainer":
        torch.manual_seed(self.cfg.seed)
        rng = np.random.default_rng(self.cfg.seed)
        seqs = self._build_sequences(train)
        if not seqs:
            log.warning("BERT4Rec: no sequences to train on.")
            return self

        n_items = len(self._item_to_token)
        device = best_device()
        self.model = BERT4Rec(n_items=n_items, cfg=self.cfg).to(device)
        ds = _MaskedSeqDataset(seqs, self.cfg, rng)
        loader = DataLoader(ds, batch_size=min(self.cfg.batch_size, len(ds)), shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        log.info("BERT4Rec training on %s", device)

        self.model.train()
        for ep in range(self.cfg.epochs):
            total, n = 0.0, 0
            for seq, lbl in loader:
                seq = seq.to(device); lbl = lbl.to(device)
                logits = self.model(seq)  # (B, L, V)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), lbl.view(-1), ignore_index=PAD_ID)
                opt.zero_grad(); loss.backward(); opt.step()
                total += float(loss.item()) * seq.size(0); n += seq.size(0)
            log.info("BERT4Rec ep %d/%d loss=%.4f", ep + 1, self.cfg.epochs, total / max(n, 1))
        return self

    # Predict next items for a user given their interaction history.
    @torch.no_grad()
    def recommend(self, history: list[int], k: int = 10, exclude_seen: bool = True) -> list[tuple[int, float]]:
        if self.model is None:
            return []
        self.model.eval()
        device = next(self.model.parameters()).device
        tokens = [self._item_to_token[int(j)] for j in history if int(j) in self._item_to_token]
        tokens = tokens[-(self.cfg.max_len - 1):] + [MASK_ID]
        seq = [PAD_ID] * (self.cfg.max_len - len(tokens)) + tokens
        x = torch.tensor([seq], dtype=torch.long, device=device)
        logits = self.model(x)[0, -1]  # predict the masked final position
        scores = logits[2:].cpu().numpy()    # skip pad/mask tokens
        # Map token back to item id.
        item_ids = np.array([self._token_to_item[i + 2] for i in range(len(scores))])
        seen = set(int(j) for j in history) if exclude_seen else set()
        mask = np.array([jid not in seen for jid in item_ids])
        scores = np.where(mask, scores, -np.inf)
        k = min(k, int(mask.sum()))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(item_ids[i]), float(scores[i])) for i in top]

    def save(self, path: Path) -> None:
        import joblib
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "bert4rec.pt")
        joblib.dump({"item_to_token": self._item_to_token, "cfg": self.cfg}, path / "bert4rec_meta.joblib")

    def load(self, path: Path) -> "BERT4RecTrainer":
        import joblib
        obj = joblib.load(path / "bert4rec_meta.joblib")
        self._item_to_token = obj["item_to_token"]
        self._token_to_item = {v: k for k, v in self._item_to_token.items()}
        self.cfg = obj["cfg"]
        device = best_device()
        self.model = BERT4Rec(n_items=len(self._item_to_token), cfg=self.cfg).to(device)
        self.model.load_state_dict(torch.load(path / "bert4rec.pt", map_location=device))
        return self
