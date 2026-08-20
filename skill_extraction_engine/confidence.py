"""
confidence.py
--------------
Per-skill confidence scoring.

Score composition (clamped to [0, 1]):

    base(match_type)
        exact           1.00  -- canonical name or listed alias, verbatim
        alias            0.95  -- synonym/abbreviation match
        fuzzy            scaled from the fuzzy match ratio (0-100 -> 0-1),
                          capped at 0.90 since it's inherently uncertain
        stack_inferred   0.65  -- inferred by expanding a skill-stack token
                          (e.g. "MERN" -> React), not stated directly

    + section_bonus      +0.05 if any mention occurred inside a section
                          conventionally associated with skills
                          ("Skills", "Certifications", "Projects")
    + frequency_bonus     +0.02 per extra mention beyond the first,
                          capped at +0.10 total

The scoring is intentionally simple and explainable (no black-box model)
so downstream ATS matching (later days) can reason about *why* a skill
scored the way it did.
"""

from __future__ import annotations

_BASE_SCORES = {
    "exact": 1.00,
    "alias": 0.95,
    "stack_inferred": 0.65,
}

_SKILL_BEARING_SECTIONS = {"skills", "certifications", "projects"}

_SECTION_BONUS = 0.05
_FREQUENCY_BONUS_PER_MENTION = 0.02
_FREQUENCY_BONUS_CAP = 0.10


def fuzzy_base_score(match_ratio: float) -> float:
    """
    Convert a rapidfuzz-style similarity ratio (0-100) into a base
    confidence score, capped below the certainty of an exact/alias match.
    """
    scaled = (match_ratio / 100.0) * 0.90
    return round(min(scaled, 0.90), 4)


def score_skill(
    match_type: str,
    mention_count: int,
    source_sections: set[str],
    fuzzy_ratio: float | None = None,
) -> float:
    """
    Compute the final confidence score for a single canonical skill,
    given how it was found, how many times it appeared, and which
    resume sections it appeared in.
    """
    if match_type == "fuzzy":
        base = fuzzy_base_score(fuzzy_ratio if fuzzy_ratio is not None else 85.0)
    else:
        base = _BASE_SCORES.get(match_type, 0.5)

    score = base

    if any(sec.lower() in _SKILL_BEARING_SECTIONS for sec in source_sections):
        score += _SECTION_BONUS

    frequency_bonus = min(
        (max(mention_count, 1) - 1) * _FREQUENCY_BONUS_PER_MENTION, _FREQUENCY_BONUS_CAP
    )
    score += frequency_bonus

    return round(max(0.0, min(score, 1.0)), 4)
