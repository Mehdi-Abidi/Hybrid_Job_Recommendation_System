"""Popularity baseline with recency decay and per-category support."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


class PopularityRecommender:
    """Job popularity = sum_over_interactions(rating * exp(-age_days / halflife)).
    Optional per-category ranking for category-aware cold-start."""

    def __init__(self, recency_halflife_days: float = 30.0):
        self.halflife = recency_halflife_days
        self.global_scores: dict[int, float] = {}
        self.per_category_scores: dict[str, dict[int, float]] = {}
        self._job_category: dict[int, str] = {}

    def fit(self, interactions: pd.DataFrame, jobs: pd.DataFrame) -> "PopularityRecommender":
        df = interactions.copy()
        age = df["timestamp_days_ago"].astype(float).to_numpy() if "timestamp_days_ago" in df.columns \
            else np.zeros(len(df))
        decay = np.exp(-np.log(2) * age / self.halflife)
        df["score"] = df["rating"].astype(float).to_numpy() * decay
        agg = df.groupby("job_id")["score"].sum()
        self.global_scores = {int(j): float(s) for j, s in agg.items()}

        if "category" in jobs.columns:
            self._job_category = {int(j): str(c) for j, c in zip(jobs["job_id"], jobs["category"])}
            df["category"] = df["job_id"].map(self._job_category)
            per_cat = df.groupby(["category", "job_id"])["score"].sum()
            for (cat, jid), s in per_cat.items():
                self.per_category_scores.setdefault(str(cat), {})[int(jid)] = float(s)

        log.info("Popularity fit: %d jobs scored, %d categories", len(self.global_scores), len(self.per_category_scores))
        return self

    def recommend(self, k: int = 10, category: str | None = None, exclude: set[int] | None = None) -> list[tuple[int, float]]:
        exclude = exclude or set()
        source = self.per_category_scores.get(category, {}) if category else self.global_scores
        items = [(j, s) for j, s in source.items() if j not in exclude]
        items.sort(key=lambda x: -x[1])
        return items[:k]

    def score_pairs(self, job_ids: list[int]) -> np.ndarray:
        return np.array([self.global_scores.get(int(j), 0.0) for j in job_ids], dtype=np.float32)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "popularity.json", "w") as f:
            json.dump({
                "halflife": self.halflife,
                "global": {str(k): v for k, v in self.global_scores.items()},
                "per_category": {c: {str(k): v for k, v in d.items()} for c, d in self.per_category_scores.items()},
                "job_category": {str(k): v for k, v in self._job_category.items()},
            }, f)

    def load(self, path: Path) -> "PopularityRecommender":
        with open(path / "popularity.json") as f:
            obj = json.load(f)
        self.halflife = obj["halflife"]
        self.global_scores = {int(k): v for k, v in obj["global"].items()}
        self.per_category_scores = {c: {int(k): v for k, v in d.items()} for c, d in obj["per_category"].items()}
        self._job_category = {int(k): v for k, v in obj["job_category"].items()}
        return self
