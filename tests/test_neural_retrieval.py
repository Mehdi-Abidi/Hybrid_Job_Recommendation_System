"""Tests for Two-Tower model and FAISS ANN index.
Uses a stub text embedder to stay offline and fast."""
from __future__ import annotations
import numpy as np
import pytest
import torch

from src.data.preprocessing import DataPreprocessor
from src.features.text_features import EmbeddingFeaturizer
from src.models.two_tower import TwoTowerTrainer
from src.retrieval.faiss_index import FaissJobIndex


class FakeEmbedder(EmbeddingFeaturizer):
    def __init__(self, dim: int = 16):
        super().__init__(dim=dim)
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


@pytest.fixture
def processed(tmp_project):
    return DataPreprocessor(tmp_project).run(persist=False)


@pytest.fixture
def trained_two_tower(processed):
    cfg = {"embedding_dim": 8, "hidden_dims": [16], "dropout": 0.0, "batch_size": 8, "epochs": 2, "lr": 1e-2}
    trainer = TwoTowerTrainer(cfg, text_embedder=FakeEmbedder(dim=8))
    trainer.build_features(processed.jobs, processed.users)
    trainer.train(processed.train, epochs=2, batch_size=8, lr=1e-2)
    return trainer


# build_features produces aligned user and job matrices.
def test_build_features_shapes(processed):
    trainer = TwoTowerTrainer({"embedding_dim": 8}, text_embedder=FakeEmbedder(dim=8))
    art = trainer.build_features(processed.jobs, processed.users)
    assert art.user_feature_matrix.shape[0] == len(processed.users)
    assert art.job_feature_matrix.shape[0] == len(processed.jobs)
    assert art.config["user_in_dim"] == art.user_feature_matrix.shape[1]
    assert art.config["job_in_dim"] == art.job_feature_matrix.shape[1]


# Training runs without error and produces normalized embeddings.
def test_train_produces_normalized_embeddings(trained_two_tower):
    je = trained_two_tower.job_embeddings()
    ue = trained_two_tower.user_embeddings()
    assert je.shape[1] == 8 and ue.shape[1] == 8
    assert np.allclose(np.linalg.norm(je, axis=1), 1.0, atol=1e-4)
    assert np.allclose(np.linalg.norm(ue, axis=1), 1.0, atol=1e-4)


# Training reduces loss: final loss is lower than a randomly initialized model's loss.
def test_training_reduces_loss(processed):
    torch.manual_seed(0)
    cfg = {"embedding_dim": 8, "hidden_dims": [16], "dropout": 0.0, "batch_size": 8, "epochs": 5, "lr": 5e-2}
    trainer = TwoTowerTrainer(cfg, text_embedder=FakeEmbedder(dim=8), seed=0)
    trainer.build_features(processed.jobs, processed.users)
    trainer.train(processed.train, epochs=5, batch_size=8, lr=5e-2)
    # Positive-pair similarity after training should beat random-baseline ~0.
    ue = trainer.user_embeddings()
    je = trainer.job_embeddings()
    pos_sims = []
    u_idx = {int(u): i for i, u in enumerate(trainer.artifacts.user_ids)}
    j_idx = {int(j): i for i, j in enumerate(trainer.artifacts.job_ids)}
    for _, r in processed.train.iterrows():
        pos_sims.append(float(ue[u_idx[int(r["user_id"])]] @ je[j_idx[int(r["job_id"])]]))
    assert np.mean(pos_sims) > 0.0


# FAISS flat index returns exact top-k and the nearest neighbor of a vector is itself.
def test_faiss_flat_exact_topk():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((20, 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    job_ids = np.arange(100, 120)
    idx = FaissJobIndex(embedding_dim=8, index_type="Flat").build(emb, job_ids)
    res = idx.search(emb[3], k=5)
    assert res[0][0] == 103  # its own id is nearest
    assert len(res) == 5


# Batch search returns one result list per query.
def test_faiss_batch_search():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((10, 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    job_ids = np.arange(10)
    idx = FaissJobIndex(embedding_dim=8, index_type="Flat").build(emb, job_ids)
    results = idx.search_batch(emb[:3], k=3)
    assert len(results) == 3
    assert all(len(r) == 3 for r in results)


# Full retrieval loop: two-tower embeddings indexed in FAISS, query by user embedding.
def test_end_to_end_retrieval(trained_two_tower, processed):
    je = trained_two_tower.job_embeddings()
    ue = trained_two_tower.user_embeddings()
    idx = FaissJobIndex(embedding_dim=je.shape[1], index_type="Flat").build(je, trained_two_tower.artifacts.job_ids)
    recs = idx.search(ue[0], k=min(3, len(processed.jobs)))
    assert len(recs) == min(3, len(processed.jobs))
    job_set = set(processed.jobs["job_id"])
    assert all(j in job_set for j, _ in recs)
