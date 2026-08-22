"""
certification_parser.py
-------------------------
Parses individual "Certifications" section lines into structured
certification records: name, issuer (if stated), year (if stated), and
a relevance category (via relevance_tagger).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from .normalizer import normalize_certification_name
from .relevance_tagger import tag_relevance

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

# Splits "AWS Certified Solutions Architect - Associate -- Amazon, 2022"
# into name / issuer using common separators. The LAST separator segment
# that isn't a bare year is treated as the issuer; everything before it
# is the certification name. Certs with no separator have name only.
_SEPARATOR_RE = re.compile(r"\s+[—\-–]{1,2}\s+")


@dataclass
class CertificationRecord:
    raw_line: str
    name: str
    issuer: Optional[str]
    year: Optional[int]
    relevance_category: str
    confidence: float


class CertificationParser:
    """Extracts CertificationRecord entries from a Certifications section."""

    def parse_section(self, certifications_text: str) -> List[CertificationRecord]:
        if not certifications_text or not certifications_text.strip():
            return []

        records: List[CertificationRecord] = []
        for raw_line in certifications_text.splitlines():
            line = raw_line.strip().lstrip("-").strip()
            if not line:
                continue
            if line.lower() in ("certifications", "certificates", "licenses & certifications"):
                continue

            # Skills-category lines ("Languages | Python, Java, SQL") are a
            # known formatting artifact from the upstream cleaner when a
            # resume's Skills section has no heading of its own and the
            # Certifications heading is the last recognized section before
            # it. They are never legitimate certification entries -- skip.
            if " | " in line:
                continue

            record = self._parse_line(line)
            if record is not None:
                records.append(record)

        return records

    # ------------------------------------------------------------------ #
    def _parse_line(self, line: str) -> Optional[CertificationRecord]:
        year = self._extract_year(line)
        without_year = _YEAR_RE.sub("", line).strip(" ,()")

        parts = _SEPARATOR_RE.split(without_year)
        parts = [p.strip(" ,") for p in parts if p.strip(" ,")]

        if not parts:
            return None

        if len(parts) == 1:
            name = parts[0]
            issuer = None
            confidence = 0.7
        else:
            # Last segment is usually the issuing body / level qualifier;
            # everything before it is the certification name.
            name = " — ".join(parts[:-1])
            issuer = parts[-1]
            confidence = 0.85

        name = normalize_certification_name(name)
        if not name:
            return None

        return CertificationRecord(
            raw_line=line,
            name=name,
            issuer=issuer or None,
            year=year,
            relevance_category=tag_relevance(name),
            confidence=confidence,
        )

    @staticmethod
    def _extract_year(text: str) -> Optional[int]:
        match = _YEAR_RE.search(text)
        return int(match.group(0)) if match else None
