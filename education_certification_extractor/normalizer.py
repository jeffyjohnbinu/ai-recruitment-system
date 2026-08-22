"""
normalizer.py
-------------
Canonicalizes degree-type abbreviations and tidies up institution names so
that the same qualification written in different ways on different resumes
("B.S.", "BS", "Bachelor of Science", "Bachelors in Science") collapses to
one normalized value downstream modules (ATS matching, screening) can rely
on without doing their own fuzzy matching.
"""

from __future__ import annotations

import re
from typing import Optional

# Longest / most specific patterns first -- matched in order, first hit wins.
# Keys are regex patterns (case-insensitive, word-boundary wrapped by caller);
# values are the canonical degree name.
_DEGREE_CANON: list[tuple[str, str]] = [
    (r"ph\.?\s*d\.?", "Doctor of Philosophy"),
    (r"doctor(ate)? of philosophy", "Doctor of Philosophy"),
    (r"m\.?\s*b\.?\s*a\.?", "Master of Business Administration"),
    (r"master'?s? of business administration", "Master of Business Administration"),
    (r"m\.?\s*tech\.?", "Master of Technology"),
    (r"master'?s? of technology", "Master of Technology"),
    (r"m\.?\s*s\.?(?!c)", "Master of Science"),
    (r"master'?s? of science", "Master of Science"),
    (r"m\.?\s*sc\.?", "Master of Science"),
    (r"m\.?\s*a\.?(?!$)", "Master of Arts"),
    (r"master'?s? of arts", "Master of Arts"),
    (r"m\.?\s*c\.?\s*a\.?", "Master of Computer Applications"),
    (r"m\.?\s*e\.?(?!ba)", "Master of Engineering"),
    (r"master'?s? of engineering", "Master of Engineering"),
    (r"b\.?\s*tech\.?", "Bachelor of Technology"),
    (r"bachelor'?s? of technology", "Bachelor of Technology"),
    (r"b\.?\s*e\.?(?!d)", "Bachelor of Engineering"),
    (r"bachelor'?s? of engineering", "Bachelor of Engineering"),
    (r"b\.?\s*s\.?(?!c)", "Bachelor of Science"),
    (r"bachelor'?s? of science", "Bachelor of Science"),
    (r"b\.?\s*sc\.?", "Bachelor of Science"),
    (r"b\.?\s*c\.?\s*a\.?", "Bachelor of Computer Applications"),
    (r"b\.?\s*a\.?(?!ch)", "Bachelor of Arts"),
    (r"bachelor'?s? of arts", "Bachelor of Arts"),
    (r"ll\.?\s*b\.?", "Bachelor of Laws"),
    (r"ll\.?\s*m\.?", "Master of Laws"),
    (r"m\.?\s*d\.?", "Doctor of Medicine"),
    (r"j\.?\s*d\.?", "Juris Doctor"),
    (r"associate'?s? (of|in) (arts|science)", "Associate Degree"),
    (r"a\.?\s*a\.?(?!$)", "Associate Degree"),
    (r"a\.?\s*s\.?(?!c)", "Associate Degree"),
    (r"diploma", "Diploma"),
    (r"high school diploma", "High School Diploma"),
]

_COMPILED_DEGREE_CANON = [(re.compile(rf"\b{p}\b", re.IGNORECASE), c) for p, c in _DEGREE_CANON]

_INSTITUTION_NOISE = re.compile(r"^(the|at)\s+", re.IGNORECASE)


def normalize_degree(raw_degree: str) -> Optional[str]:
    """
    Map a raw degree token/phrase (e.g. "B.S.", "Bachelors", "M.Tech")
    to a canonical degree name. Returns None if nothing recognizable
    is found so callers can flag low-confidence records instead of
    silently guessing.
    """
    if not raw_degree:
        return None
    candidate = raw_degree.strip()
    for pattern, canonical in _COMPILED_DEGREE_CANON:
        if pattern.search(candidate):
            return canonical
    return None


def normalize_institution(raw_institution: str) -> str:
    """Trim leading articles/prepositions and stray punctuation/whitespace."""
    if not raw_institution:
        return ""
    cleaned = raw_institution.strip(" \t-–—,.")
    cleaned = _INSTITUTION_NOISE.sub("", cleaned).strip()
    return cleaned


def normalize_certification_name(raw_name: str) -> str:
    """Tidy whitespace/punctuation around a certification title."""
    if not raw_name:
        return ""
    cleaned = re.sub(r"\s{2,}", " ", raw_name.strip(" \t-–—,."))
    return cleaned
