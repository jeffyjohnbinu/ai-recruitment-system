"""
Date & Duration Utilities
--------------------------
Parses the free-text date tokens found in resume experience lines
("2021", "Jan 2021", "January 2021", "Present", "Current") into
(year, month) tuples and computes durations in months between them.

Design notes:
  - A bare year ("2021") is treated as January of that year for the
    purpose of a *start* date and December of that year for an *end*
    date, so single-year ranges still produce a sensible duration.
  - "Present" / "Current" / "Now" resolve to the current calendar
    month at parse time (`today()` is injectable for deterministic
    tests).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Optional, Tuple

YearMonth = Tuple[int, int]  # (year, month) both 1-indexed

_MONTH_NAMES = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

_PRESENT_TOKENS = {"present", "current", "currently", "now", "ongoing", "till date", "to date"}

_YEAR_ONLY_RE = re.compile(r"^(\d{4})$")
_MONTH_YEAR_RE = re.compile(r"^([A-Za-z]{3,9})\.?\s+(\d{4})$")
_YEAR_MONTH_RE = re.compile(r"^(\d{4})[/\-]\s?(\d{1,2})$")


def is_present_token(token: str) -> bool:
    return token.strip().lower() in _PRESENT_TOKENS


def parse_date_token(
    token: str, *, is_end: bool = False, today: Optional[date] = None
) -> Optional[YearMonth]:
    """
    Parse a single date token into a (year, month) tuple.

    Returns None only if the token is unparseable. "Present"-style
    tokens resolve to the current (year, month) rather than None, so
    downstream duration math doesn't need special-casing.
    """
    if token is None:
        return None
    cleaned = token.strip().strip(".")
    if not cleaned:
        return None

    if is_present_token(cleaned):
        ref = today or date.today()
        return (ref.year, ref.month)

    m = _MONTH_YEAR_RE.match(cleaned)
    if m:
        month_name, year = m.group(1).lower(), int(m.group(2))
        month = _MONTH_NAMES.get(month_name)
        if month:
            return (year, month)

    m = _YEAR_MONTH_RE.match(cleaned)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return (year, month)

    m = _YEAR_ONLY_RE.match(cleaned)
    if m:
        year = int(m.group(1))
        return (year, 12 if is_end else 1)

    return None


def months_between(start: YearMonth, end: YearMonth) -> int:
    """
    Inclusive month count between two (year, month) tuples.
    E.g. Jan 2020 -> Mar 2020 = 3 months. Never returns a negative
    value (clamped to 0 for malformed/reversed ranges).
    """
    total = (end[0] - start[0]) * 12 + (end[1] - start[1]) + 1
    return max(total, 0)


@dataclass
class ParsedDuration:
    start: Optional[YearMonth]
    end: Optional[YearMonth]
    months: int
    is_current: bool
    parse_ok: bool


def parse_duration(
    start_token: str, end_token: str, *, today: Optional[date] = None
) -> ParsedDuration:
    is_current = is_present_token(end_token)
    start = parse_date_token(start_token, is_end=False, today=today)
    end = parse_date_token(end_token, is_end=True, today=today)

    if start is None or end is None:
        return ParsedDuration(start=start, end=end, months=0, is_current=is_current, parse_ok=False)

    months = months_between(start, end)
    return ParsedDuration(start=start, end=end, months=months, is_current=is_current, parse_ok=True)


def yearmonth_to_iso(ym: Optional[YearMonth]) -> Optional[str]:
    if ym is None:
        return None
    return f"{ym[0]:04d}-{ym[1]:02d}"
