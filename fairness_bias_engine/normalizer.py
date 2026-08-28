"""
normalizer.py
-------------
Standardizes heterogeneous candidate profiles (assembled from Day 5-11
outputs: resume extraction, skill extraction, experience parsing,
education parsing) into one canonical schema so every candidate is
compared on the same footing, regardless of resume formatting, template,
or verbosity.

Why this matters for fairness:
    Two candidates with identical qualifications can look very different
    to a scorer purely because of formatting choices (a two-column resume,
    a skills table vs. a skills paragraph, "5 yrs" vs "five years", a
    fancy template vs. a plain one). None of that is job-relevant signal.
    Normalizing structure BEFORE scoring removes formatting itself as an
    accidental source of bias.

Design notes:
    - Alias-tolerant: accepts multiple upstream key spellings (schema
      drift between days is expected and absorbed here), per project
      convention (see Day 7 metadata envelope + Day 9-11 loaders).
    - Additive-only: this module does not mutate or replace upstream
      records; it produces a new CanonicalCandidateProfile alongside them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "fairness-normalizer-1.0"
PIPELINE_VERSION = "day15"

# Canonical degree ladder, most-common variant spellings first.
_DEGREE_TIERS = [
    ("PhD", ["phd", "ph.d", "doctorate", "doctoral"]),
    (
        "Masters",
        ["masters", "master's", "m.s", "ms ", "msc", "m.tech", "mtech", "mba", "m.a", "ma "],
    ),
    (
        "Bachelors",
        [
            "bachelors",
            "bachelor's",
            "b.s",
            "bs ",
            "bsc",
            "b.tech",
            "btech",
            "b.a",
            "ba ",
            "b.e",
            "be ",
        ],
    ),
    ("Associate", ["associate degree", "a.a", "a.s", "diploma"]),
]

_WHITESPACE_RE = re.compile(r"\s+")
_YEARS_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:\+)?\s*year")


def _get_alias(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present, non-None value among several possible key spellings."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _clean_skill(skill: str) -> str:
    s = _WHITESPACE_RE.sub(" ", skill.strip().lower())
    return s


@dataclass
class CanonicalCandidateProfile:
    candidate_id: str
    standardized_skills: List[str] = field(default_factory=list)
    total_experience_years: float = 0.0
    education_level: str = "Unknown"
    certifications: List[str] = field(default_factory=list)
    normalized_titles: List[str] = field(default_factory=list)
    source_field_count: int = 0
    normalization_notes: List[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
    model_version: str = MODEL_VERSION
    pipeline_version: str = PIPELINE_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "standardized_skills": self.standardized_skills,
            "total_experience_years": self.total_experience_years,
            "education_level": self.education_level,
            "certifications": self.certifications,
            "normalized_titles": self.normalized_titles,
            "source_field_count": self.source_field_count,
            "normalization_notes": self.normalization_notes,
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "pipeline_version": self.pipeline_version,
        }


class ResumeNormalizer:
    """
    Converts a raw, heterogeneous candidate record (a merge of Day 5-11
    outputs, or any dict-shaped equivalent) into a CanonicalCandidateProfile.
    """

    def normalize(self, raw_profile: Dict[str, Any]) -> CanonicalCandidateProfile:
        notes: List[str] = []

        candidate_id = str(
            _get_alias(raw_profile, "candidate_id", "id", "source_file", default="unknown")
        )

        skills = self._normalize_skills(raw_profile, notes)
        experience_years = self._normalize_experience(raw_profile, notes)
        education_level = self._normalize_education(raw_profile, notes)
        certifications = self._normalize_certifications(raw_profile)
        titles = self._normalize_titles(raw_profile)

        return CanonicalCandidateProfile(
            candidate_id=candidate_id,
            standardized_skills=skills,
            total_experience_years=experience_years,
            education_level=education_level,
            certifications=certifications,
            normalized_titles=titles,
            source_field_count=len(raw_profile),
            normalization_notes=notes,
        )

    # ------------------------------------------------------------------ #
    def _normalize_skills(self, raw: Dict[str, Any], notes: List[str]) -> List[str]:
        raw_skills = _get_alias(
            raw, "standardized_skills", "skills", "extracted_skills", "skill_list", default=[]
        )
        if isinstance(raw_skills, dict):
            # some upstream shapes group skills by category: {"Languages": [...], ...}
            flattened: List[str] = []
            for group in raw_skills.values():
                if isinstance(group, list):
                    flattened.extend(group)
            raw_skills = flattened
            notes.append("flattened category-grouped skills dict")

        if not isinstance(raw_skills, list):
            return []

        cleaned = sorted({_clean_skill(str(s)) for s in raw_skills if str(s).strip()})
        return cleaned

    def _normalize_experience(self, raw: Dict[str, Any], notes: List[str]) -> float:
        years = _get_alias(
            raw, "total_experience_years", "experience_years", "years_of_experience", default=None
        )
        if years is not None:
            try:
                return round(float(years), 2)
            except (TypeError, ValueError):
                notes.append(f"could not parse experience years value: {years!r}")

        # Fall back to summing explicit experience record durations if present.
        records = _get_alias(raw, "experience_records", "experience", default=[])
        if isinstance(records, list) and records:
            total_months = 0.0
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                months = _get_alias(rec, "duration_months", "months", default=None)
                if months is not None:
                    try:
                        total_months += float(months)
                    except (TypeError, ValueError):
                        continue
            if total_months:
                notes.append("derived total experience from summed record durations")
                return round(total_months / 12.0, 2)

        # Last resort: scan free text for "N years" mentions.
        text_blob = _get_alias(raw, "cleaned_text", "summary", "raw_text", default="")
        if isinstance(text_blob, str) and text_blob:
            match = _YEARS_NUMBER_RE.search(text_blob.lower())
            if match:
                notes.append("derived experience years from free-text mention")
                return round(float(match.group(1)), 2)

        return 0.0

    def _normalize_education(self, raw: Dict[str, Any], notes: List[str]) -> str:
        degree_text = _get_alias(raw, "education_level", "highest_degree", "degree", default="")
        if not degree_text:
            academic = _get_alias(raw, "academic_profile", "education", default=None)
            if isinstance(academic, dict):
                degree_text = _get_alias(academic, "highest_degree", "degree", default="")
            elif isinstance(academic, list) and academic:
                degree_text = (
                    _get_alias(academic[0], "degree", default="")
                    if isinstance(academic[0], dict)
                    else ""
                )

        degree_text = str(degree_text or "").strip().lower()
        if not degree_text:
            return "Unknown"

        for tier, variants in _DEGREE_TIERS:
            if any(v in degree_text for v in variants):
                return tier

        notes.append(f"unrecognized degree text left as 'Other': {degree_text!r}")
        return "Other"

    def _normalize_certifications(self, raw: Dict[str, Any]) -> List[str]:
        certs = _get_alias(raw, "certifications", "certs", default=[])
        if not isinstance(certs, list):
            return []
        cleaned = []
        for c in certs:
            if isinstance(c, dict):
                name = _get_alias(c, "name", "title", default=None)
                if name:
                    cleaned.append(str(name).strip())
            else:
                text = str(c).strip()
                if text:
                    cleaned.append(text)
        return sorted(set(cleaned))

    def _normalize_titles(self, raw: Dict[str, Any]) -> List[str]:
        records = _get_alias(raw, "experience_records", "experience", default=[])
        titles: List[str] = []
        if isinstance(records, list):
            for rec in records:
                if isinstance(rec, dict):
                    title = _get_alias(rec, "title", "job_title", "role", default=None)
                    if title:
                        titles.append(str(title).strip())
        return titles
