"""Shared logger factory."""
import logging
import sys


# Return a configured module-level logger.
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s | %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    logger.setLevel(level)
    logger.propagate = False
    return logger
