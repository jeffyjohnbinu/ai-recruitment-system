"""
Field Extractors
------------------
Pulls structured fields out of cleaned job-description text:
  - role title (+ normalized role, + seniority level)
  - required vs. preferred skills (normalized via synonyms.py)
  - experience requirements (min/max years)
  - education preferences (degree level + field of study)

Each extractor is a small, independently testable function that takes
cleaned text (and, where useful, section-scoped text) and returns a
plain structure. `extractors.py` intentionally has no knowledge of file
I/O — that lives in parser.py / storage.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .synonyms import SENIORITY_KEYWORDS, role_lookup, skill_lookup

# --------------------------------------------------------------------- #
# Role extraction
# --------------------------------------------------------------------- #

# A role title is usually the first non-empty line, or a line explicitly
# labeled "Job Title:" / "Position:" / "Role:".
_TITLE_LABEL_PATTERN = re.compile(
    r"^\s*(job title|position|role|title)\s*[:\-]\s*(.+)$", re.IGNORECASE
)


@dataclass
class RoleInfo:
    raw_title: Optional[str]
    normalized_role: Optional[str]
    seniority_level: Optional[str]


def extract_role(cleaned_text: str) -> RoleInfo:
    lines = [ln.strip() for ln in cleaned_text.splitlines() if ln.strip()]

    raw_title = None
    for line in lines[:15]:
        m = _TITLE_LABEL_PATTERN.match(line)
        if m:
            raw_title = m.group(2).strip()
            break

    if raw_title is None and lines:
        # Fall back to the first substantive line, provided it doesn't
        # look like a canonical section heading itself and isn't a long
        # sentence (title lines are short).
        for line in lines[:5]:
            if line in {
                "Role Overview",
                "Responsibilities",
                "Requirements",
                "Preferred Qualifications",
                "Skills",
                "Education",
                "Experience",
                "Benefits",
                "Company",
            }:
                continue
            if len(line.split()) <= 8:
                raw_title = line
                break

    normalized_role = _normalize_role(raw_title) if raw_title else None
    seniority_level = _detect_seniority(raw_title, cleaned_text)

    return RoleInfo(
        raw_title=raw_title,
        normalized_role=normalized_role,
        seniority_level=seniority_level,
    )


def _normalize_role(raw_title: str) -> Optional[str]:
    lookup = role_lookup()
    title_lower = raw_title.lower()

    # Direct alias containment match, longest alias first for specificity
    for alias in sorted(lookup.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", title_lower):
            return lookup[alias]
    return None


def _detect_seniority(raw_title: Optional[str], cleaned_text: str) -> Optional[str]:
    search_space = (raw_title or "") + "\n" + "\n".join(cleaned_text.splitlines()[:20])
    search_lower = search_space.lower()

    # Check most-senior-first so "Senior Manager" resolves to Manager only
    # if "senior" isn't also a stronger signal; simple first-match wins,
    # ordered by specificity.
    for level in ["Executive", "Manager", "Staff", "Senior", "Entry-Level", "Mid-Level", "Intern"]:
        for keyword in SENIORITY_KEYWORDS[level]:
            if re.search(rf"\b{re.escape(keyword.strip())}\b", search_lower):
                return level
    return None


# --------------------------------------------------------------------- #
# Skill extraction
# --------------------------------------------------------------------- #


@dataclass
class SkillsInfo:
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    unmatched_terms: List[str] = field(default_factory=list)


def extract_skills(cleaned_text: str, required_section: str, preferred_section: str) -> SkillsInfo:
    """
    Normalize skill mentions into canonical form. Skills found in a
    "Preferred Qualifications" section are kept separate from those in
    "Requirements" / "Skills" so callers can weight them differently.
    Skills mentioned in the full document but outside either scoped
    section still count as required (most JDs list core skills inline
    without a dedicated section).
    """
    lookup = skill_lookup()

    required_scope = required_section if required_section.strip() else cleaned_text
    preferred_scope = preferred_section

    required = _match_skills(required_scope, lookup)
    preferred = _match_skills(preferred_scope, lookup)

    # Don't double-count a skill as both required and preferred
    preferred = [s for s in preferred if s not in required]

    return SkillsInfo(required_skills=required, preferred_skills=preferred)


def _match_skills(text: str, lookup: dict) -> List[str]:
    if not text.strip():
        return []
    text_lower = f" {text.lower()} "
    found = []
    seen = set()
    for alias in sorted(lookup.keys(), key=len, reverse=True):
        pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
        if re.search(pattern, text_lower):
            canonical = lookup[alias]
            if canonical not in seen:
                seen.add(canonical)
                found.append(canonical)
    return found


# --------------------------------------------------------------------- #
# Experience extraction
# --------------------------------------------------------------------- #


@dataclass
class ExperienceInfo:
    min_years: Optional[float] = None
    max_years: Optional[float] = None
    raw_mentions: List[str] = field(default_factory=list)


_EXPERIENCE_PATTERNS = [
    # "3-5 years", "3 to 5 years"
    re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|to|–)\s*(\d+(?:\.\d+)?)\+?\s*years?", re.IGNORECASE),
    # "5+ years", "minimum of 5 years", "at least 5 years"
    re.compile(r"(?:minimum of|at least|over)?\s*(\d+(?:\.\d+)?)\+\s*years?", re.IGNORECASE),
    # plain "5 years of experience"
    re.compile(r"(\d+(?:\.\d+)?)\s*years?(?:\s+of)?\s+experience", re.IGNORECASE),
]


def extract_experience(cleaned_text: str) -> ExperienceInfo:
    info = ExperienceInfo()
    text = cleaned_text

    range_match = _EXPERIENCE_PATTERNS[0].search(text)
    if range_match:
        lo, hi = float(range_match.group(1)), float(range_match.group(2))
        info.min_years, info.max_years = min(lo, hi), max(lo, hi)
        info.raw_mentions.append(range_match.group(0).strip())
        return info

    plus_match = _EXPERIENCE_PATTERNS[1].search(text)
    if plus_match:
        info.min_years = float(plus_match.group(1))
        info.max_years = None
        info.raw_mentions.append(plus_match.group(0).strip())
        return info

    plain_match = _EXPERIENCE_PATTERNS[2].search(text)
    if plain_match:
        info.min_years = float(plain_match.group(1))
        info.max_years = None
        info.raw_mentions.append(plain_match.group(0).strip())

    return info


# --------------------------------------------------------------------- #
# Education extraction
# --------------------------------------------------------------------- #


@dataclass
class EducationInfo:
    degree_level: Optional[str] = None
    fields_of_study: List[str] = field(default_factory=list)
    raw_mentions: List[str] = field(default_factory=list)


_DEGREE_LEVEL_PATTERNS = [
    ("PhD", re.compile(r"\b(ph\.?d\.?|doctorate|doctoral)\b", re.IGNORECASE)),
    ("Master's", re.compile(r"\b(master'?s?|m\.?s\.?|m\.?b\.?a\.?|m\.?tech\.?)\b", re.IGNORECASE)),
    (
        "Bachelor's",
        re.compile(
            r"\b(bachelor'?s?|b\.?s\.?|b\.?a\.?|b\.?tech\.?|undergraduate degree)\b",
            re.IGNORECASE,
        ),
    ),
    ("Associate", re.compile(r"\b(associate'?s? degree)\b", re.IGNORECASE)),
    ("High School", re.compile(r"\b(high school diploma|ged)\b", re.IGNORECASE)),
]

_FIELD_OF_STUDY_KEYWORDS = [
    "computer science",
    "software engineering",
    "information technology",
    "data science",
    "statistics",
    "mathematics",
    "electrical engineering",
    "computer engineering",
    "business administration",
    "finance",
    "economics",
    "marketing",
    "design",
    "information systems",
    "engineering",
    "physics",
]


_FIELD_WINDOW_CHARS = 90  # search radius around a degree mention for the field of study


def extract_education(cleaned_text: str, education_section: str) -> EducationInfo:
    """
    Detect degree level, then look for a field-of-study keyword only in a
    window of text right after the degree mention (e.g. "Bachelor's degree
    *in Computer Science*"). Restricting the search window — rather than
    scanning the whole requirements section — avoids false positives like
    matching "Engineering" inside an unrelated phrase such as "software
    engineering experience" elsewhere in the same section.
    """
    info = EducationInfo()
    scope = education_section if education_section.strip() else cleaned_text

    degree_match = None
    for level, pattern in _DEGREE_LEVEL_PATTERNS:
        m = pattern.search(scope)
        if m:
            info.degree_level = level
            info.raw_mentions.append(m.group(0).strip())
            degree_match = m
            break

    if degree_match:
        window = scope[degree_match.end() : degree_match.end() + _FIELD_WINDOW_CHARS].lower()
    else:
        window = scope.lower()

    for field_kw in _FIELD_OF_STUDY_KEYWORDS:
        if field_kw in window:
            info.fields_of_study.append(field_kw.title())

    return info
