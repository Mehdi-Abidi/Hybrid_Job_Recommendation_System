"""Verify the two-tower feature matrix is bounded — location no longer blows up to 10k+ dims."""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.models.two_tower import TwoTowerTrainer
from src.features.text_features import EmbeddingFeaturizer


class _StubEmbedder(EmbeddingFeaturizer):
    """Skip the heavy MiniLM load — return deterministic 8-dim vectors."""
    def __init__(self):
        super().__init__(model_name="stub", dim=8, batch_size=4)
    def encode(self, texts, normalize=True):
        out = np.zeros((len(texts), 8), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in str(t).split():
                out[i, hash(tok) % 8] += 1.0
            n = np.linalg.norm(out[i])
            if normalize and n > 0:
                out[i] /= n
        return out


def test_job_feature_dim_is_bounded():
    """With 1000 unique location strings (worst case), job_feat must stay small."""
    n_locs = 1000
    jobs = pd.DataFrame({
        "job_id": np.arange(n_locs),
        "title": [f"j{i}" for i in range(n_locs)],
        "description": ["d"] * n_locs,
        "category": ["software"] * n_locs,
        "skills": ["python"] * n_locs,
        "location": [f"City{i}, State{i % 50}, Country{i % 5}" for i in range(n_locs)],
    })
    users = pd.DataFrame({
        "user_id": [0],
        "skills": ["python"],
        "experience_years": [5],
        "preferred_location": ["City1, State1, Country0"],
        "resume_text": ["resume"],
    })
    trainer = TwoTowerTrainer({"embedding_dim": 8, "hidden_dims": [16], "dropout": 0.0,
                               "epochs": 1, "batch_size": 4}, text_embedder=_StubEmbedder())
    art = trainer.build_features(jobs, users)
    # Skills (~1) + cat one-hot (~2) + 256 loc buckets + ~50 states + ~5 countries + 8 text
    # ≪ 1000 (raw location one-hot) — total should be < 350.
    assert art.job_feature_matrix.shape[1] < 350, (
        f"two-tower job_feat too wide: {art.job_feature_matrix.shape[1]}"
    )
    # Sanity: locations were hashed, not one-hot.
    assert art.loc_hash.n_buckets == 256
