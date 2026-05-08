"""Career path prediction: given a user's current title + experience, suggest next roles.
Uses transition statistics mined from the interaction corpus (proxy: apply-graph over job titles)."""
from __future__ import annotations
from collections import defaultdict, Counter
from dataclasses import dataclass
import pandas as pd


@dataclass
class CareerStep:
    from_title: str
    to_title: str
    probability: float


class CareerPathPredictor:
    """Learns title→title transitions from per-user sequences of applied jobs."""

    def __init__(self, apply_rating_threshold: int = 5):
        self.threshold = apply_rating_threshold
        self.transitions: dict[str, Counter] = defaultdict(Counter)

    def fit(self, interactions: pd.DataFrame, jobs: pd.DataFrame) -> "CareerPathPredictor":
        if "title" not in jobs.columns:
            return self
        j2title = {int(j): str(t).lower() for j, t in zip(jobs["job_id"], jobs["title"])}
        applied = interactions[interactions["rating"] >= self.threshold].copy()
        sort_col = "timestamp_days_ago" if "timestamp_days_ago" in applied.columns else None
        for _, g in applied.groupby("user_id"):
            if sort_col:
                g = g.sort_values(sort_col, ascending=False)  # oldest first
            titles = [j2title.get(int(j), "") for j in g["job_id"]]
            for a, b in zip(titles, titles[1:]):
                if a and b and a != b:
                    self.transitions[a][b] += 1
        return self

    def predict_next(self, current_title: str, k: int = 5) -> list[CareerStep]:
        current = str(current_title).lower().strip()
        counts = self.transitions.get(current)
        if not counts:
            return []
        total = sum(counts.values())
        return [CareerStep(current, t, c / total) for t, c in counts.most_common(k)]
