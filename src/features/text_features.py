"""TF-IDF and sentence-transformer embedding featurizers for job/user text."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from src.utils.logging import get_logger

log = get_logger(__name__)


# TF-IDF over job text (title + description + skills).
class TFIDFFeaturizer:
    def __init__(self, max_features: int = 5000, ngram_range: tuple[int, int] = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range,
            stop_words="english", lowercase=True, min_df=2,
        )
        self.fitted = False

    def fit(self, texts: Iterable[str]) -> "TFIDFFeaturizer":
        self.vectorizer.fit(list(texts))
        self.fitted = True
        return self

    def transform(self, texts: Iterable[str]) -> csr_matrix:
        return self.vectorizer.transform(list(texts))

    def fit_transform(self, texts: Iterable[str]) -> csr_matrix:
        m = self.vectorizer.fit_transform(list(texts))
        self.fitted = True
        return m

    def save(self, path: Path) -> None:
        joblib.dump(self.vectorizer, path)

    def load(self, path: Path) -> "TFIDFFeaturizer":
        self.vectorizer = joblib.load(path)
        self.fitted = True
        return self


# Sentence-transformer dense embeddings; lazy model load.
class EmbeddingFeaturizer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 384, batch_size: int = 64):
        self.model_name = model_name
        self.dim = dim
        self.batch_size = batch_size
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            log.info("Loading sentence-transformer: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        model = self._load_model()
        emb = model.encode(
            texts, batch_size=self.batch_size, convert_to_numpy=True,
            normalize_embeddings=normalize, show_progress_bar=False,
        )
        return emb.astype(np.float32)

    @staticmethod
    def save_embeddings(emb: np.ndarray, path: Path) -> None:
        np.save(path, emb)

    @staticmethod
    def load_embeddings(path: Path) -> np.ndarray:
        return np.load(path)


# Compose the canonical text representation for a job row.
def job_text(row) -> str:
    return f"{row['title']}. {row['description']} Skills: {row['skills']}"


# Compose the canonical text representation for a user row.
def user_text(row) -> str:
    return f"{row['resume_text']} Skills: {row['skills']}"
