"""
utils/logger.py
----------------
Centralized logging for the entire AI recruitment system.

Usage in any module:

    from utils.logger import get_logger
    logger = get_logger(__name__)

    logger.info("Parsed resume for candidate %s", candidate_id)
    logger.warning("Low match score: %.2f", score)
    logger.error("Failed to parse file: %s", filepath, exc_info=True)

Design:
- One rotating file per day under logs/, plus console output.
- Every module gets its own named logger (module path), but all
  loggers share the same handlers/formatting, configured once.
- Log level and directory are controlled via config/settings.py
  (which itself reads from .env), so behavior differs cleanly
  between development and production without code changes.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from config.settings import settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure_root_logger() -> None:
    """Configure the root logger exactly once per process."""
    global _configured
    if _configured:
        return

    log_dir: Path = settings.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger("ai_recruitment_system")
    root_logger.setLevel(settings.log_level.upper())
    root_logger.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler — human-readable output during development
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Rotating file handler — one file per day, kept for 14 days
    file_handler = TimedRotatingFileHandler(
        filename=log_dir / "app.log",
        when="midnight",
        backupCount=14,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Separate file capturing only errors/criticals for fast triage
    error_handler = logging.FileHandler(log_dir / "errors.log", encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a namespaced logger (e.g. "ai_recruitment_system.parsers.resume_parser")
    that inherits handlers from the configured root logger.
    """
    _configure_root_logger()
    return logging.getLogger(f"ai_recruitment_system.{name}")
