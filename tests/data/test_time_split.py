"""Verify the time-based split has no temporal leakage.

`timestamp_days_ago` is large for old, small for recent. After a global temporal
split with the older portion as train, every test interaction must be at least
as recent as the newest train interaction — i.e. test has SMALLER days_ago.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import pytest

from src.data.preprocessing import _split_time_based


def test_time_based_split_has_no_temporal_leakage():
    # 100 interactions spanning days_ago in [0..99]
    df = pd.DataFrame({
        "user_id": list(range(100)),
        "job_id": list(range(100)),
        "rating": [1] * 100,
        "timestamp_days_ago": list(range(100)),
    })
    train, test = _split_time_based(df, test_size=0.2)
    # Train = older = larger days_ago. Test = newer = smaller days_ago.
    assert train["timestamp_days_ago"].min() >= test["timestamp_days_ago"].max(), (
        f"leakage: train_min={train['timestamp_days_ago'].min()} "
        f"test_max={test['timestamp_days_ago'].max()}"
    )
    # Sanity: ~80/20 split
    assert 75 <= len(train) <= 85
    assert 15 <= len(test) <= 25


def test_default_strategy_is_time_based():
    """Config default must be `time_based` so the production training run uses it."""
    from config import load_settings
    cfg = load_settings()
    assert cfg.split.strategy == "time_based", (
        f"expected default time_based, got {cfg.split.strategy}"
    )
