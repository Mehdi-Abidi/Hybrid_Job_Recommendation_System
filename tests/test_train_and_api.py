"""Integration tests for the training orchestrator and API endpoints.
Uses the FakeEmbedder to stay offline; exercises the full fit → save → load → API path."""
from __future__ import annotations
from unittest.mock import patch
import numpy as np
import pytest

from src.features.text_features import EmbeddingFeaturizer


class FakeEmbedder(EmbeddingFeaturizer):
    def __init__(self, model_name: str = "fake", dim: int = 16, batch_size: int = 64):
        super().__init__(model_name=model_name, dim=dim, batch_size=batch_size)
        self.dim = dim

    def encode(self, texts, normalize=True):
        rng = np.random.default_rng(0)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        vocab: dict[str, np.ndarray] = {}
        for i, t in enumerate(texts):
            for tok in str(t).lower().split():
                if tok not in vocab:
                    vocab[tok] = rng.standard_normal(self.dim).astype(np.float32)
                out[i] += vocab[tok]
        if normalize:
            out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
        return out


# Train orchestrator end-to-end on tiny synthetic data; all classical+neural+LTR fit and save.
def test_train_pipeline_end_to_end(tmp_project, monkeypatch):
    # Shrink configs so training completes in seconds.
    tmp_project.models["two_tower"] = {
        "embedding_dim": 8, "hidden_dims": [16], "dropout": 0.0,
        "batch_size": 4, "epochs": 2, "lr": 1e-2, "negatives_per_positive": 2,
    }
    tmp_project.models["ltr"] = {
        "objective": "rank:ndcg", "n_estimators": 5,
        "learning_rate": 0.1, "max_depth": 2,
    }
    tmp_project.faiss = {"index_type": "Flat", "nlist": 1, "nprobe": 1}

    # Patch EmbeddingFeaturizer to offline fake.
    with patch("src.models.train.EmbeddingFeaturizer", FakeEmbedder):
        from src.models.train import run
        out = run(tmp_project, include_tier2=False)

    assert out["content"] is not None
    assert out["collab"] is not None
    assert out["two_tower"].model is not None
    assert out["faiss"].index is not None
    assert "report" in out and not out["report"].empty

    # Artifacts were saved.
    root = tmp_project.path("artifacts")
    assert (root / "content_based" / "job_embeddings.npy").exists()
    assert (root / "collaborative" / "ials.npz").exists()
    assert (root / "popularity" / "popularity.json").exists()
    assert (root / "two_tower" / "two_tower.pt").exists()
    assert (root / "faiss" / "jobs.faiss").exists()
    assert (root / "ltr" / "ltr.json").exists()


# API import smoke test: the module loads and registers the expected routes.
def test_api_imports_and_routes_registered():
    from api.main import app
    paths = {r.path for r in app.routes}
    for expected in ["/health", "/recommend/{user_id}", "/recommend/new-user",
                     "/similar-jobs/{job_id}", "/explain/{user_id}/{job_id}",
                     "/recommend/multi-stage/{user_id}", "/pipeline/inspect/{user_id}",
                     "/query/parse", "/salary/predict", "/skill-gap"]:
        assert expected in paths, f"missing route: {expected}"
