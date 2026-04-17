"""Simple logging setup."""
from __future__ import annotations

import logging
import os
import sys


def setup_logging(level: str | None = None) -> logging.Logger:
    level = level or os.environ.get("LOG_LEVEL", "INFO")
    root = logging.getLogger()
    if root.handlers:
        return logging.getLogger("phase2")
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(h)
    root.setLevel(level)
    return logging.getLogger("phase2")
