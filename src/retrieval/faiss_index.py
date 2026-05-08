"""FAISS ANN index over job embeddings — inner product (assumes L2-normalized vectors)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import faiss

from src.utils.logging import get_logger

log = get_logger(__name__)


class FaissJobIndex:
    """Wraps a FAISS index mapping job embeddings -> job_ids.
    index_type: 'Flat' (exact IP) or 'IVFFlat' (approx, trains with nlist centroids)."""

    def __init__(self, embedding_dim: int = 128, index_type: str = "IVFFlat",
                 nlist: int = 100, nprobe: int = 10):
        self.dim = embedding_dim
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.index: faiss.Index | None = None
        self.job_ids: np.ndarray | None = None

    def build(self, job_embeddings: np.ndarray, job_ids: np.ndarray) -> "FaissJobIndex":
        assert job_embeddings.shape[1] == self.dim
        emb = np.ascontiguousarray(job_embeddings, dtype=np.float32)
        if self.index_type == "Flat" or len(job_ids) < max(self.nlist * 2, 32):
            # Fall back to exact index for small corpora where IVF training is unstable.
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(emb)
        else:
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(emb)
            self.index.add(emb)
            self.index.nprobe = self.nprobe
        self.job_ids = np.asarray(job_ids)
        log.info("FAISS built: %s, dim=%d, n=%d", self.index_type, self.dim, len(job_ids))
        return self

    # k-NN by inner product. Returns (job_id, score) tuples.
    def search(self, user_embedding: np.ndarray, k: int = 500) -> list[tuple[int, float]]:
        assert self.index is not None and self.job_ids is not None
        q = np.ascontiguousarray(user_embedding.reshape(1, -1), dtype=np.float32)
        k = min(k, len(self.job_ids))
        scores, idx = self.index.search(q, k)
        return [(int(self.job_ids[i]), float(s)) for i, s in zip(idx[0], scores[0]) if i != -1]

    def search_batch(self, user_embeddings: np.ndarray, k: int = 500) -> list[list[tuple[int, float]]]:
        assert self.index is not None and self.job_ids is not None
        q = np.ascontiguousarray(user_embeddings, dtype=np.float32)
        k = min(k, len(self.job_ids))
        scores, idx = self.index.search(q, k)
        return [[(int(self.job_ids[j]), float(s)) for j, s in zip(idx[r], scores[r]) if j != -1]
                for r in range(len(user_embeddings))]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "jobs.faiss"))
        np.save(path / "job_ids.npy", self.job_ids)

    def load(self, path: Path) -> "FaissJobIndex":
        self.index = faiss.read_index(str(path / "jobs.faiss"))
        self.job_ids = np.load(path / "job_ids.npy")
        if self.index_type == "IVFFlat" and hasattr(self.index, "nprobe"):
            self.index.nprobe = self.nprobe
        return self
