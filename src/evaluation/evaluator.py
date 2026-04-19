"""Evaluator: run recommenders against test holdouts and compute metrics.

A "recommender" is any object/function exposing `recommend(user_id, k) -> list[(job_id, score)]`
or a plain callable of that signature. For the multi-stage pipeline, the adapter
returns just the job_ids from Recommendation objects.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any
import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    precision_at_k, recall_at_k, ndcg_at_k, average_precision,
    coverage, intra_list_diversity, mean_metric,
)
from src.utils.logging import get_logger

log = get_logger(__name__)


# Callable that returns a ranked list of (job_id, score) for (user_id, k).
RecommendFn = Callable[[int, int], list[tuple[int, float]]]


@dataclass
class EvaluationReport:
    model_name: str
    k_values: list[int]
    precision: dict[int, float]
    recall: dict[int, float]
    ndcg: dict[int, float]
    map_score: float
    coverage: float
    diversity: float
    n_users_evaluated: int

    def as_row(self) -> dict[str, Any]:
        row = {"model": self.model_name, "n_users": self.n_users_evaluated,
               "map": self.map_score, "coverage": self.coverage, "diversity": self.diversity}
        for k in self.k_values:
            row[f"precision@{k}"] = self.precision[k]
            row[f"recall@{k}"] = self.recall[k]
            row[f"ndcg@{k}"] = self.ndcg[k]
        return row


class Evaluator:
    def __init__(self, test: pd.DataFrame, k_values: list[int], jobs: pd.DataFrame,
                 item_embeddings: np.ndarray | None = None,
                 id_to_row: dict[int, int] | None = None,
                 positive_rating_threshold: int = 3):
        self.test = test
        self.k_values = k_values
        self.jobs = jobs
        self.item_embeddings = item_embeddings
        self.id_to_row = id_to_row or {}
        self.threshold = positive_rating_threshold
        self._relevant_by_user: dict[int, set[int]] = {
            int(u): {int(j) for j, r in zip(g["job_id"], g["rating"]) if r >= self.threshold}
            for u, g in test.groupby("user_id")
        }
        self.n_items = len(jobs)

    def evaluate(self, recommend_fn: RecommendFn, model_name: str,
                 user_ids: list[int] | None = None) -> EvaluationReport:
        users = user_ids if user_ids is not None else list(self._relevant_by_user.keys())
        max_k = max(self.k_values)
        per_user_prec = {k: [] for k in self.k_values}
        per_user_rec = {k: [] for k in self.k_values}
        per_user_ndcg = {k: [] for k in self.k_values}
        per_user_map = []
        all_recs: list[list[int]] = []
        diversities: list[float] = []

        for uid in users:
            relevant = self._relevant_by_user.get(int(uid), set())
            if not relevant:
                continue
            ranked = recommend_fn(int(uid), max_k)
            rec_ids = [jid for jid, _ in ranked]
            all_recs.append(rec_ids)
            for k in self.k_values:
                per_user_prec[k].append(precision_at_k(rec_ids, relevant, k))
                per_user_rec[k].append(recall_at_k(rec_ids, relevant, k))
                per_user_ndcg[k].append(ndcg_at_k(rec_ids, relevant, k))
            per_user_map.append(average_precision(rec_ids, relevant, max_k))
            if self.item_embeddings is not None and self.id_to_row:
                diversities.append(intra_list_diversity(rec_ids, self.item_embeddings, self.id_to_row))

        report = EvaluationReport(
            model_name=model_name, k_values=self.k_values,
            precision={k: mean_metric(per_user_prec[k]) for k in self.k_values},
            recall={k: mean_metric(per_user_rec[k]) for k in self.k_values},
            ndcg={k: mean_metric(per_user_ndcg[k]) for k in self.k_values},
            map_score=mean_metric(per_user_map),
            coverage=coverage(all_recs, self.n_items),
            diversity=mean_metric(diversities) if diversities else 0.0,
            n_users_evaluated=len(per_user_map),
        )
        log.info("Eval %s: n=%d, ndcg@%d=%.3f, map=%.3f",
                 model_name, report.n_users_evaluated, max_k,
                 report.ndcg[max_k], report.map_score)
        return report

    # Run a dict of recommenders and return a DataFrame.
    def compare_models(self, models: dict[str, RecommendFn],
                       user_ids: list[int] | None = None) -> pd.DataFrame:
        rows = [self.evaluate(fn, name, user_ids).as_row() for name, fn in models.items()]
        return pd.DataFrame(rows)

    # Per-interaction-count-bucket metrics: how does a model handle cold vs. warm users?
    def cold_start_analysis(self, recommend_fn: RecommendFn, model_name: str,
                            train: pd.DataFrame, buckets: tuple[tuple[int, int], ...] = (
                                (0, 0), (1, 4), (5, 14), (15, 10_000))) -> pd.DataFrame:
        counts = train.groupby("user_id").size().to_dict()
        rows = []
        for lo, hi in buckets:
            bucket_users = [int(u) for u in self._relevant_by_user
                            if lo <= int(counts.get(int(u), 0)) <= hi]
            if not bucket_users:
                rows.append({"model": model_name, "bucket": f"{lo}-{hi}", "n_users": 0})
                continue
            rep = self.evaluate(recommend_fn, model_name, user_ids=bucket_users)
            row = rep.as_row()
            row["bucket"] = f"{lo}-{hi}"
            rows.append(row)
        return pd.DataFrame(rows)
