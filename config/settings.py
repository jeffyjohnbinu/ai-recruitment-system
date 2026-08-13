"""
config/settings.py
-------------------
Single source of truth for application configuration.
All modules should import `settings` from here rather than reading
os.environ directly, so config stays centralized and testable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load variables from .env into the process environment (no-op if absent)
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    # --- App ---
    app_env: str = os.getenv("APP_ENV", "development")
    base_dir: Path = field(default_factory=lambda: BASE_DIR)

    # --- Logging ---
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_dir: Path = field(default_factory=lambda: BASE_DIR / os.getenv("LOG_DIR", "logs"))

    # --- AI provider ---
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    ai_model: str = os.getenv("AI_MODEL", "gpt-4o-mini")

    # --- ATS ---
    ats_min_match_score: float = float(os.getenv("ATS_MIN_MATCH_SCORE", "0.65"))

    # --- Data paths ---
    data_raw_dir: Path = field(default_factory=lambda: BASE_DIR / "data" / "raw")
    data_processed_dir: Path = field(default_factory=lambda: BASE_DIR / "data" / "processed")

    def is_production(self) -> bool:
        return self.app_env == "production"


settings = Settings()

# Ensure critical directories exist at import time
settings.log_dir.mkdir(parents=True, exist_ok=True)
settings.data_raw_dir.mkdir(parents=True, exist_ok=True)
settings.data_processed_dir.mkdir(parents=True, exist_ok=True)
