"""Typed config loader — Pydantic-free, minimal."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


# Dataclasses mirror config.yaml sections.
@dataclass
class SyntheticCfg:
    n_users: int = 1000
    n_jobs: int = 5000
    n_interactions: int = 20000
    seed: int = 42


@dataclass
class DataCfg:
    raw_path: str
    processed_path: str
    jobs_file: str
    users_file: str
    interactions_file: str
    kaggle_dataset: str
    use_synthetic: bool
    synthetic: SyntheticCfg


@dataclass
class SplitCfg:
    test_size: float = 0.2
    strategy: str = "per_user"
    seed: int = 42


@dataclass
class Settings:
    data: DataCfg
    ratings: dict[str, int]
    split: SplitCfg
    models: dict[str, Any]
    faiss: dict[str, Any]
    evaluation: dict[str, Any]
    paths: dict[str, str]
    project_root: Path

    # Resolve a configured path relative to the project root.
    def path(self, key: str) -> Path:
        if key == "raw":
            return self.project_root / self.data.raw_path
        if key == "processed":
            return self.project_root / self.data.processed_path
        if key == "artifacts":
            return self.project_root / self.paths["artifacts"]
        raise KeyError(key)


# Load and return a Settings object from config.yaml.
def load_settings(config_path: str | Path | None = None) -> Settings:
    project_root = Path(__file__).resolve().parent.parent
    cfg_path = Path(config_path) if config_path else project_root / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data = DataCfg(
        raw_path=raw["data"]["raw_path"],
        processed_path=raw["data"]["processed_path"],
        jobs_file=raw["data"]["jobs_file"],
        users_file=raw["data"]["users_file"],
        interactions_file=raw["data"]["interactions_file"],
        kaggle_dataset=raw["data"]["kaggle_dataset"],
        use_synthetic=raw["data"]["use_synthetic"],
        synthetic=SyntheticCfg(**raw["data"]["synthetic"]),
    )
    split = SplitCfg(**raw["split"])

    return Settings(
        data=data,
        ratings=raw["ratings"],
        split=split,
        models=raw["models"],
        faiss=raw["faiss"],
        evaluation=raw["evaluation"],
        paths=raw["paths"],
        project_root=project_root,
    )
