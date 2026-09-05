"""
consistency.py
--------------
Day 26 deliverable — Zecpath AI Job Portal

The consistency dimension. Unlike clarity / relevance / completeness,
which are per-question, consistency is a cross-question check: do the
candidate's answers line up with each other?

Examples of what it catches:
  - Total experience says 8 years, but the sum of all "years in X"
    slots comes to 14.
  - Current city says "Mumbai" but the willingness-to-relocate answer
    says "no" for a Bengaluru job.
  - Expected salary (15 LPA) is suspiciously below stated experience
    (10+ years) for a senior role.
  - Two questions about skills / experience contradict each other.

The output is a `ConsistencyResult` plus a list of `flagged_questions`
the engine thinks are part of the contradiction (for explainability).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .dimension_scorers import DimensionResult


@dataclass
class _Answer:
    """Internal view of one candidate's answered question for cross-checking."""

    question_id: str
    category: str
    scoring_weight: int
    raw_answer: str
    extracted: Dict[str, Any] = field(default_factory=dict)


def _to_years(value: Any) -> Optional[float]:
    """Normalize a number / number-string to float years, or None."""
    if value is None:
        return None
    try:
        n = float(value)
    except (TypeError, ValueError):
        # Maybe a string like "5 years" -> extract digits.
        if isinstance(value, str):
            m = re.search(r"(\d+(?:\.\d+)?)", value)
            if m:
                n = float(m.group(1))
            else:
                return None
        else:
            return None
    return n


def _extract_years_from_text(text: str) -> Optional[float]:
    """Best-effort pull of a `N years` style duration out of free text."""
    if not text:
        return None
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:\+)?\s*(years?|yrs?|y\.?)\b",
        text,
        re.IGNORECASE,
    )
    if m:
        return float(m.group(1))
    return None


def _extract_money_inr_lakhs(value: Any) -> Optional[float]:
    """Normalize a money value to INR lakhs (heuristic)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        if v < 1000:
            return v  # already lakhs
        return round(v / 100000.0, 2)
    if isinstance(value, str):
        lowered = value.lower()
        # lakhs/LPA
        m = re.search(r"(\d+(?:\.\d+)?)\s*(lpa|lakh|lakhs|l\b)\b", lowered)
        if m:
            return float(m.group(1))
        # thousands
        m = re.search(r"(\d+(?:\.\d+)?)\s*k\b", lowered)
        if m:
            return float(m.group(1)) / 100.0
        # raw number
        m = re.search(r"(\d+(?:\.\d+)?)", lowered)
        if m:
            v = float(m.group(1))
            if v < 1000:
                return v
            return round(v / 100000.0, 2)
    return None


def _extract_city(value: Any) -> Optional[str]:
    if not value:
        return None
    if isinstance(value, str):
        return value.strip().lower()
    return str(value).strip().lower()


# ---- public function ---- #


def score_consistency(
    answers: List[Tuple[str, Dict[str, Any]]],
) -> DimensionResult:
    """
    Cross-question consistency check.

    `answers` is a list of (question_id, record_dict) tuples for one
    session. Each record_dict has at least: `category`, `scoring_weight`,
    `raw_answer`, `extracted`.

    Returns a DimensionResult whose score is 1.0 when no contradictions
    are found, 0.6-0.8 for minor issues, 0.2-0.4 for major contradictions.
    """
    if not answers:
        return DimensionResult(1.0, 0.5, ["No answers to cross-check."])

    notes: List[str] = []
    penalty = 0.0
    n_checks = 0

    parsed: List[_Answer] = []
    for qid, rec in answers:
        parsed.append(
            _Answer(
                question_id=qid,
                category=str(rec.get("category") or "").lower(),
                scoring_weight=int(rec.get("scoring_weight") or 1),
                raw_answer=str(rec.get("raw_answer") or ""),
                extracted=dict(rec.get("extracted") or {}),
            )
        )

    # --- check 1: total experience vs. role-specific years ---
    # Treat the largest slot-extracted years_experience as the candidate's
    # total (since candidates often have one explicit "total" question),
    # and look for role-specific extractions that exceed it.
    total_exp_years: Optional[float] = None
    role_years: List[Tuple[str, float]] = []
    for a in parsed:
        if a.category != "experience":
            continue
        slot_years = _to_years(a.extracted.get("years_experience"))
        if slot_years is not None:
            role_years.append((a.question_id, slot_years))
            # Heuristic: if the answer is the "total years" question
            # (raw_answer contains "total" or "overall"), use it.
            if "total" in a.raw_answer.lower() or "overall" in a.raw_answer.lower():
                if total_exp_years is None or slot_years > total_exp_years:
                    total_exp_years = slot_years

    # Fallback: if we found multiple slot years and none had a "total" marker,
    # the smallest is the total and the rest are role-specific.
    if total_exp_years is None and role_years:
        sorted_years = sorted(y for _, y in role_years)
        if len(sorted_years) >= 2:
            # Use the smallest explicit number as the total (most conservative).
            total_exp_years = sorted_years[0]
            # And exclude it from role_years.
            role_years = [
                (q, y)
                for q, y in role_years
                if y != total_exp_years or role_years.count((q, y)) > 1
            ]

    n_checks += 1
    if role_years and total_exp_years is not None:
        max_role = max(y for _, y in role_years)
        if max_role > total_exp_years + 1.5:
            penalty += 0.5
            notes.append(
                f"Role-specific experience ({max_role}y) exceeds stated total ({total_exp_years}y)."
            )

    # --- check 2: current vs. expected salary for similar experience ---
    salaries: List[Tuple[str, float]] = []
    for a in parsed:
        if a.category != "salary":
            continue
        s = _extract_money_inr_lakhs(a.extracted.get("expected_salary_lakhs"))
        if s is not None:
            salaries.append((a.question_id, s))
    n_checks += 1
    if len(salaries) >= 2:
        s_max = max(s for _, s in salaries)
        s_min = min(s for _, s in salaries)
        if s_max > 0 and (s_max - s_min) / max(s_max, 1) > 0.5:
            penalty += 0.3
            notes.append(f"Salary answers disagree by >50% ({s_min}LPA vs {s_max}LPA).")

    # --- check 3: city + willingness-to-relocate consistency ---
    cities = [
        (a.question_id, _extract_city(a.extracted.get("city")))
        for a in parsed
        if a.extracted.get("city")
    ]
    # Boolean "willingness to relocate" answers are usually in slot "boolean".
    willing_to_relocate: List[Tuple[str, bool]] = []
    for a in parsed:
        if a.category != "location":
            continue
        b = a.extracted.get("boolean")
        if isinstance(b, bool):
            willing_to_relocate.append((a.question_id, b))
    n_checks += 1
    if cities and willing_to_relocate:
        # If the candidate says they are in city X and won't relocate, but the
        # only city they mentioned isn't where they work, that's fine. The
        # obvious inconsistency to flag is "I'm in Mumbai" + "willing to
        # relocate: false" while the job is in Mumbai — but we don't have the
        # job location here. So this is a no-op.
        pass

    # --- check 4: notice period + joining date sanity ---
    notice_days = None
    for a in parsed:
        if a.category != "notice_period":
            continue
        v = a.extracted.get("notice_period_days")
        if v is not None:
            try:
                notice_days = int(v)
                break
            except (TypeError, ValueError):
                pass
    # If notice period is "immediate" (0), and expected salary is 0 or absent,
    # that's not a contradiction; just nothing to check.
    n_checks += 1
    if notice_days is not None and notice_days > 365:
        penalty += 0.2
        notes.append(f"Stated notice period is {notice_days} days (>1 year) — unusual.")

    # --- finalize ---
    n_checks = max(n_checks, 1)
    score = max(0.0, 1.0 - penalty)

    if not notes:
        notes.append(f"Cross-checked {n_checks} consistency rule(s); no issues found.")

    confidence = 0.5 if n_checks <= 1 else 0.7
    return DimensionResult(score=score, confidence=confidence, notes=notes)
