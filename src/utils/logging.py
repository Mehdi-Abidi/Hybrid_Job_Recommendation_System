"""Shared logger factory."""
import logging
import sys


# Force stdout to UTF-8 once at import time. On Windows the default `cp1252`
# encoding crashes the logging handler whenever any unicode (arrows, math
# symbols) is emitted — and crashed handlers silently freeze the parent.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


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
