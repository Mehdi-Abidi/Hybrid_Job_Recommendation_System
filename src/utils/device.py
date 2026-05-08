"""Pick GPU if available, else CPU. One call, used by every neural model."""
from __future__ import annotations
import torch


def best_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
