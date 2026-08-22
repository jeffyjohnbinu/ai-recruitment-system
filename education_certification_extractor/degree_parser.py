"""
degree_parser.py
-----------------
Parses individual "Education" section lines into structured degree
records: degree type, field of study, institution, graduation year.

Two-pass strategy (same pattern used in Day 8's section segmentation):
  Pass 1 -- rule-based regex extraction, which handles the large majority
            of resume education lines ("B.S. in Computer Science --
            University of Texas at Austin, 2018").
  Pass 2 -- looser fallback heuristics for lines that don't match the
            common template, so we still return a partial record (e.g.
            institution + year only) instead of dropping the line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from .normalizer import normalize_degree, normalize_institution

# A degree token is any of the common abbreviations/spelled-out forms.
# Matched loosely here; normalize_degree() does the canonicalization.
_DEGREE_TOKEN = (
    r"(Ph\.?\s*D\.?|Doctorate|M\.?\s*B\.?\s*A\.?|M\.?\s*Tech\.?|M\.?\s*S\.?c?\.?|"
    r"M\.?\s*A\.?|M\.?\s*C\.?\s*A\.?|M\.?\s*E\.?|LL\.?\s*M\.?|LL\.?\s*B\.?|M\.?\s*D\.?|"
    r"J\.?\s*D\.?|B\.?\s*Tech\.?|B\.?\s*E\.?|B\.?\s*S\.?c?\.?|B\.?\s*A\.?|B\.?\s*C\.?\s*A\.?|"
    r"Associate'?s?|Diploma|Bachelor'?s?(?:\s+(?:of|in))?|Master'?s?(?:\s+(?:of|in))?)"
)

# e.g. "B.S. in Computer Science — University of Texas at Austin, 2018"
_FULL_LINE_RE = re.compile(
    rf"^{_DEGREE_TOKEN}\.?\s*(?:(?:of|in)\s+([A-Za-z0-9&,/ \-]+?))?"
    rf"\s*(?:[—\-–]|@|\bat\b|\bfrom\b)\s*"
    rf"(?P<institution>[A-Za-z0-9&.,'’ \-]+?)"
    rf"[,\s]*\(?(?P<year>(19|20)\d{{2}})\)?\s*$",
    re.IGNORECASE,
)

# Looser fallback: degree + field, no explicit institution/year on this line
_DEGREE_FIELD_RE = re.compile(
    rf"^{_DEGREE_TOKEN}\.?\s*(?:(?:of|in)\s+(?P<field>[A-Za-z0-9&,/ \-]+))?",
    re.IGNORECASE,
)

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_YEAR_RANGE_RE = re.compile(
    r"\b((19|20)\d{2})\s*[-–—to]+\s*((19|20)\d{2}|present)\b", re.IGNORECASE
)


@dataclass
class DegreeRecord:
    raw_line: str
    degree_type_raw: Optional[str]
    degree_type_normalized: Optional[str]
    field_of_study: Optional[str]
    institution: Optional[str]
    graduation_year: Optional[int]
    confidence: float  # 0.0-1.0, based on which pass/pattern matched


class DegreeParser:
    """Extracts DegreeRecord entries from an Education section's text."""

    def parse_section(self, education_text: str) -> List[DegreeRecord]:
        if not education_text or not education_text.strip():
            return []

        records: List[DegreeRecord] = []
        for raw_line in education_text.splitlines():
            line = raw_line.strip().lstrip("-").strip()
            if not line:
                continue
            # Skip the section heading itself if it slipped in
            if line.lower() in ("education", "academic background", "academic qualifications"):
                continue

            record = self._parse_line(line)
            if record is not None:
                records.append(record)

        return records

    # ------------------------------------------------------------------ #
    def _parse_line(self, line: str) -> Optional[DegreeRecord]:
        # Pass 1: full structured match (degree + field + institution + year)
        match = _FULL_LINE_RE.match(line)
        if match:
            degree_raw = match.group(1)
            field = (match.group(2) or "").strip(" ,-") or None
            institution = normalize_institution(match.group("institution"))
            year = int(match.group("year"))
            return DegreeRecord(
                raw_line=line,
                degree_type_raw=degree_raw,
                degree_type_normalized=normalize_degree(degree_raw),
                field_of_study=field,
                institution=institution or None,
                graduation_year=year,
                confidence=0.95,
            )

        # Pass 2: degree token present, but line doesn't fit the full template.
        # Salvage whatever we can (degree/field always; institution/year via
        # loose scanning of the remainder of the line).
        loose_match = _DEGREE_FIELD_RE.match(line)
        if loose_match:
            degree_raw = loose_match.group(1)
            field = (loose_match.group("field") or "").strip(" ,-") or None
            remainder = line[loose_match.end() :].strip(" ,-–—")

            year = self._extract_year(remainder) or self._extract_year(line)
            institution = self._extract_institution_fallback(remainder)

            return DegreeRecord(
                raw_line=line,
                degree_type_raw=degree_raw,
                degree_type_normalized=normalize_degree(degree_raw),
                field_of_study=field,
                institution=institution,
                graduation_year=year,
                confidence=0.6,
            )

        # No recognizable degree token at all. If the line at least looks
        # like an institution + year (common on a second line under a
        # degree header), surface it as a low-confidence, degree-less
        # record rather than silently dropping it.
        year = self._extract_year(line)
        if year and len(line) < 120:
            institution = self._extract_institution_fallback(line)
            if institution:
                return DegreeRecord(
                    raw_line=line,
                    degree_type_raw=None,
                    degree_type_normalized=None,
                    field_of_study=None,
                    institution=institution,
                    graduation_year=year,
                    confidence=0.3,
                )

        return None

    @staticmethod
    def _extract_year(text: str) -> Optional[int]:
        range_match = _YEAR_RANGE_RE.search(text)
        if range_match:
            end = range_match.group(3)
            if end and end.isdigit():
                return int(end)
            start = range_match.group(1)
            return int(start) if start else None
        year_match = _YEAR_RE.search(text)
        return int(year_match.group(0)) if year_match else None

    @staticmethod
    def _extract_institution_fallback(text: str) -> Optional[str]:
        # Strip a trailing year/date range, leftover separators, then
        # whatever's left (if anything substantive) is treated as the
        # institution name.
        stripped = _YEAR_RANGE_RE.sub("", text)
        stripped = _YEAR_RE.sub("", stripped)
        stripped = normalize_institution(stripped)
        return stripped or None
