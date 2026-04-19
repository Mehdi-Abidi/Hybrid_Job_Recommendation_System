"""Multi-hot skill encoding, categorical encoding, experience binning."""
from __future__ import annotations
import numpy as np
import pandas as pd


# Parse a "a,b,c" skills cell to a set of lowercased skill tokens.
def parse_skills(cell: str) -> set[str]:
    if not isinstance(cell, str) or not cell.strip():
        return set()
    return {s.strip().lower() for s in cell.split(",") if s.strip()}


class SkillEncoder:
    """Fit a fixed skill vocabulary from training jobs+users, transform to multi-hot."""

    def __init__(self):
        self.vocab: list[str] = []
        self.index: dict[str, int] = {}

    def fit(self, skill_cells: list[str]) -> "SkillEncoder":
        vocab: set[str] = set()
        for c in skill_cells:
            vocab.update(parse_skills(c))
        self.vocab = sorted(vocab)
        self.index = {s: i for i, s in enumerate(self.vocab)}
        return self

    def transform(self, skill_cells: list[str]) -> np.ndarray:
        out = np.zeros((len(skill_cells), len(self.vocab)), dtype=np.float32)
        for i, c in enumerate(skill_cells):
            for s in parse_skills(c):
                j = self.index.get(s)
                if j is not None:
                    out[i, j] = 1.0
        return out

    def overlap_count(self, a: str, b: str) -> int:
        return len(parse_skills(a) & parse_skills(b))


class CategoricalEncoder:
    """Dense integer encoding with an OOV bucket at index 0."""

    def __init__(self):
        self.classes_: list[str] = []
        self.index_: dict[str, int] = {}

    def fit(self, values: list[str]) -> "CategoricalEncoder":
        uniq = sorted({str(v) for v in values})
        self.classes_ = ["<OOV>"] + uniq
        self.index_ = {v: i for i, v in enumerate(self.classes_)}
        return self

    def transform(self, values: list[str]) -> np.ndarray:
        return np.array([self.index_.get(str(v), 0) for v in values], dtype=np.int64)

    @property
    def size(self) -> int:
        return len(self.classes_)


# Bucket years of experience into seniority bins.
def bin_experience(years: pd.Series) -> np.ndarray:
    bins = [-1, 1, 3, 6, 10, 100]  # junior: 0-1, mid: 2-3, senior: 4-6, staff: 7-10, principal: 11+
    return pd.cut(years, bins=bins, labels=False).astype(np.int64).to_numpy()
