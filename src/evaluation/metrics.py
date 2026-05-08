"""Ranking metrics: Precision@K, Recall@K, NDCG@K, MAP, coverage, diversity.

All functions accept:
  recommended: list[int] — ranked list of job_ids (best first)
  relevant:    set[int]  — ground-truth relevant job_ids
"""
from __future__ import annotations
from collections.abc import Iterable
import numpy as np


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if k <= 0 or not recommended:
        return 0.0
    top = recommended[:k]
    return sum(1 for j in top if j in relevant) / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top = recommended[:k]
    return sum(1 for j in top if j in relevant) / len(relevant)


# DCG using the common relevance-as-gain convention (binary relevance).
def dcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    top = recommended[:k]
    return float(sum(1.0 / np.log2(i + 2) for i, j in enumerate(top) if j in relevant))


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    ideal = min(len(relevant), k)
    if ideal == 0:
        return 0.0
    idcg = float(sum(1.0 / np.log2(i + 2) for i in range(ideal)))
    return dcg_at_k(recommended, relevant, k) / idcg


def average_precision(recommended: list[int], relevant: set[int], k: int | None = None) -> float:
    if not relevant:
        return 0.0
    top = recommended if k is None else recommended[:k]
    hits = 0
    total = 0.0
    for i, j in enumerate(top, start=1):
        if j in relevant:
            hits += 1
            total += hits / i
    return total / min(len(relevant), len(top) or 1)


# Coverage: fraction of the item catalog ever recommended across all users.
def coverage(all_recommendations: Iterable[list[int]], n_items: int) -> float:
    seen: set[int] = set()
    for rec in all_recommendations:
        seen.update(rec)
    return len(seen) / max(n_items, 1)


# Intra-list diversity: 1 - mean pairwise cosine similarity of recommended items.
def intra_list_diversity(recommended: list[int], item_embeddings: np.ndarray,
                         id_to_row: dict[int, int]) -> float:
    idx = [id_to_row[j] for j in recommended if j in id_to_row]
    if len(idx) < 2:
        return 0.0
    v = item_embeddings[idx]
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    v = v / n
    sim = v @ v.T
    iu = np.triu_indices(len(idx), k=1)
    return float(1.0 - sim[iu].mean())


# Batch helpers: average a per-user metric over many users.
def mean_metric(per_user: list[float]) -> float:
    arr = [x for x in per_user if x is not None]
    return float(np.mean(arr)) if arr else 0.0
