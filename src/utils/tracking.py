"""Thin MLflow wrapper. Falls back to no-op if mlflow unavailable or MLFLOW_DISABLE=1.

Keeps tests independent of mlflow and lets us turn tracking off via env var on
constrained environments (e.g. Colab smoke runs)."""
from __future__ import annotations
from contextlib import contextmanager
from typing import Any, Iterator
import os

try:
    import mlflow as _mlflow  # type: ignore
    _HAS_MLFLOW = True
except ImportError:
    _mlflow = None
    _HAS_MLFLOW = False


def _enabled() -> bool:
    return _HAS_MLFLOW and os.getenv("MLFLOW_DISABLE") != "1"


@contextmanager
def run(experiment: str, run_name: str | None = None) -> Iterator[Any]:
    if not _enabled():
        yield None
        return
    _mlflow.set_experiment(experiment)
    with _mlflow.start_run(run_name=run_name) as r:
        yield r


def log_params(d: dict[str, Any]) -> None:
    if _enabled():
        _mlflow.log_params({k: str(v) for k, v in d.items()})


def log_metrics(d: dict[str, float], step: int | None = None) -> None:
    if _enabled():
        _mlflow.log_metrics({k: float(v) for k, v in d.items()}, step=step)


def log_artifact(path: str) -> None:
    if _enabled():
        _mlflow.log_artifact(path)
