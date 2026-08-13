"""
utils/validators.py
--------------------
Small, dependency-free validation helpers shared across parsers,
ats_engine, and scoring modules.
"""

from __future__ import annotations

import re

_EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}$")
_PHONE_RE = re.compile(r"^\+?[0-9\s\-()]{7,15}$")


def is_valid_email(value: str) -> bool:
    """Return True if `value` looks like a syntactically valid email address."""
    return bool(value) and bool(_EMAIL_RE.match(value.strip()))


def is_valid_phone(value: str) -> bool:
    """Return True if `value` looks like a syntactically valid phone number."""
    return bool(value) and bool(_PHONE_RE.match(value.strip()))


def clamp_score(score: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a numeric score into the [low, high] range."""
    return max(low, min(high, score))
