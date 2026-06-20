"""
Centralized logging configuration for the Market Regime Detection project.

Provides a factory function `get_logger()` that returns a configured logger
with both console (stdout) and file output. Log files are stored in the
project root under `logs/`.
"""

import logging
import os
from datetime import datetime


# Resolve project root (two levels up from src/utils/)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_LOG_DIR = os.path.join(_PROJECT_ROOT, "logs")


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create and return a named logger with console and file handlers.

    Args:
        name:  Logger name (typically __name__ of the calling module).
        level: Logging level (default: INFO).

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # ── Formatter ──────────────────────────────────────────────
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── File handler ───────────────────────────────────────────
    os.makedirs(_LOG_DIR, exist_ok=True)
    log_filename = f"market_regime_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(
        os.path.join(_LOG_DIR, log_filename), encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
