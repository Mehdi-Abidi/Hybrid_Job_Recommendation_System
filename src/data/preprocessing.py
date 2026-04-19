"""Load raw CSVs, clean, assign implicit ratings, build sparse matrix, train/test split."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz

from config import Settings, load_settings
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ProcessedData:
    jobs: pd.DataFrame
    users: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    interaction_matrix: csr_matrix     # (n_users, n_jobs), train-only
    user_index: dict[int, int]         # user_id -> row index
    job_index: dict[int, int]          # job_id -> col index


# Strip/normalize job text fields and drop invalid rows.
def _clean_jobs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["job_id"]).copy()
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["description"] = df["description"].fillna(df["title"]).astype(str).str.strip()
    df["location"] = df["location"].fillna("Unknown").astype(str).str.strip()
    df["skills"] = df["skills"].fillna("").astype(str)
    df = df[df["title"] != ""].reset_index(drop=True)
    return df


# Normalize user fields and parse skills.
def _clean_users(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["user_id"]).copy()
    df["skills"] = df["skills"].fillna("").astype(str)
    df["resume_text"] = df["resume_text"].fillna("").astype(str)
    df["experience_years"] = df["experience_years"].fillna(0).astype("int64").clip(lower=0)
    return df.reset_index(drop=True)


# Assign implicit ratings per action and keep the strongest action per (user, job).
def _clean_interactions(df: pd.DataFrame, ratings: dict[str, int],
                        valid_users: set[int], valid_jobs: set[int]) -> pd.DataFrame:
    df = df.copy()
    df = df[df["user_id"].isin(valid_users) & df["job_id"].isin(valid_jobs)]
    df["rating"] = df["action"].map(ratings).fillna(1).astype(int)
    # Keep the strongest signal per (user, job) pair.
    df = df.sort_values("rating", ascending=False).drop_duplicates(["user_id", "job_id"]).reset_index(drop=True)
    return df


# Per-user random split: each user's interactions split independently into train/test.
def _split_per_user(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for _, g in df.groupby("user_id"):
        idx = g.index.to_numpy()
        if len(idx) < 2:
            train_idx.extend(idx)      # Keep single-interaction users in train only
            continue
        n_test = max(1, int(round(len(idx) * test_size)))
        perm = rng.permutation(idx)
        test_idx.extend(perm[:n_test])
        train_idx.extend(perm[n_test:])
    return df.loc[train_idx].reset_index(drop=True), df.loc[test_idx].reset_index(drop=True)


# Time-based split using timestamp_days_ago (smaller = more recent -> test).
def _split_time_based(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("timestamp_days_ago", ascending=False).reset_index(drop=True)  # oldest first
    cut = int(len(df) * (1 - test_size))
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


# Build CSR user-job matrix with implicit ratings as values.
def _build_matrix(train: pd.DataFrame, user_index: dict[int, int], job_index: dict[int, int]) -> csr_matrix:
    rows = train["user_id"].map(user_index).to_numpy()
    cols = train["job_id"].map(job_index).to_numpy()
    data = train["rating"].astype(np.float32).to_numpy()
    return csr_matrix((data, (rows, cols)), shape=(len(user_index), len(job_index)))


class DataPreprocessor:
    """Full preprocessing pipeline: load → clean → split → persist."""

    def __init__(self, cfg: Settings):
        self.cfg = cfg

    # Load raw CSVs.
    def load_raw(self) -> dict[str, pd.DataFrame]:
        raw = self.cfg.path("raw")
        return {
            "jobs": pd.read_csv(raw / self.cfg.data.jobs_file),
            "users": pd.read_csv(raw / self.cfg.data.users_file),
            "interactions": pd.read_csv(raw / self.cfg.data.interactions_file),
        }

    # Run full pipeline and return in-memory structures (caller decides whether to persist).
    def run(self, persist: bool = True) -> ProcessedData:
        raw = self.load_raw()
        jobs = _clean_jobs(raw["jobs"])
        users = _clean_users(raw["users"])
        interactions = _clean_interactions(
            raw["interactions"], self.cfg.ratings,
            set(users["user_id"]), set(jobs["job_id"]),
        )

        strategy: Literal["per_user", "time_based"] = self.cfg.split.strategy  # type: ignore
        if strategy == "time_based" and "timestamp_days_ago" in interactions.columns:
            train, test = _split_time_based(interactions, self.cfg.split.test_size)
        else:
            train, test = _split_per_user(interactions, self.cfg.split.test_size, self.cfg.split.seed)

        user_index = {uid: i for i, uid in enumerate(users["user_id"].to_numpy())}
        job_index = {jid: i for i, jid in enumerate(jobs["job_id"].to_numpy())}
        matrix = _build_matrix(train, user_index, job_index)

        log.info("Preprocessing done: %d jobs, %d users, %d train / %d test interactions, sparsity=%.4f",
                 len(jobs), len(users), len(train), len(test),
                 1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1]))

        data = ProcessedData(
            jobs=jobs, users=users, train=train, test=test,
            interaction_matrix=matrix, user_index=user_index, job_index=job_index,
        )
        if persist:
            self.save(data)
        return data

    # Persist processed artifacts.
    def save(self, data: ProcessedData) -> None:
        p = self.cfg.path("processed")
        p.mkdir(parents=True, exist_ok=True)
        data.jobs.to_csv(p / "jobs.csv", index=False)
        data.users.to_csv(p / "users.csv", index=False)
        data.train.to_csv(p / "train.csv", index=False)
        data.test.to_csv(p / "test.csv", index=False)
        save_npz(p / "interaction_matrix.npz", data.interaction_matrix)
        import json
        with open(p / "indices.json", "w") as f:
            json.dump({
                "user_index": {str(k): v for k, v in data.user_index.items()},
                "job_index": {str(k): v for k, v in data.job_index.items()},
            }, f)
        log.info("Persisted processed data to %s", p)

    # Load previously processed artifacts.
    def load(self) -> ProcessedData:
        p = self.cfg.path("processed")
        import json
        with open(p / "indices.json") as f:
            idx = json.load(f)
        return ProcessedData(
            jobs=pd.read_csv(p / "jobs.csv"),
            users=pd.read_csv(p / "users.csv"),
            train=pd.read_csv(p / "train.csv"),
            test=pd.read_csv(p / "test.csv"),
            interaction_matrix=load_npz(p / "interaction_matrix.npz"),
            user_index={int(k): v for k, v in idx["user_index"].items()},
            job_index={int(k): v for k, v in idx["job_index"].items()},
        )


# CLI entrypoint.
def main() -> None:
    cfg = load_settings()
    DataPreprocessor(cfg).run()


if __name__ == "__main__":
    main()
