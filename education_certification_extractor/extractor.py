"""
Education & Certification Parsing Engine
------------------------------------------
Top-level orchestrator: resume text/sections -> degree parser ->
certification parser -> normalization -> structured academic profile.

Usage (pre-segmented sections, e.g. from Day 8's resume_section_classifier):
    engine = EducationCertificationExtractor(output_dir="outputs")
    record = engine.extract_from_sections(
        {"Education": "...", "Certifications": "..."},
        source_file="john_doe.docx",
    )

Usage (raw cleaned text, e.g. from Day 5's ResumeExtractionEngine):
    record = engine.extract_from_text(cleaned_text, source_file="john_doe.pdf")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from .certification_parser import CertificationParser
from .degree_parser import DegreeParser
from .storage import SCHEMA_VERSION, AcademicProfileRecord, ResultStore

logger = logging.getLogger("education_certification_extractor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Canonical section names + aliases, kept consistent with Day 5's
# cleaner._SECTION_ALIASES so this module recognizes the same headings
# without importing resume_extraction_engine directly.
_EDUCATION_ALIASES = [
    "education",
    "academic background",
    "academic qualifications",
    "educational qualifications",
]
_CERTIFICATION_ALIASES = [
    "certifications",
    "certificates",
    "licenses & certifications",
    "licenses and certifications",
]

_ALL_KNOWN_HEADINGS = {
    "summary",
    "professional summary",
    "career summary",
    "profile",
    "about me",
    "objective",
    "experience",
    "work experience",
    "professional experience",
    "employment history",
    "work history",
    "skills",
    "technical skills",
    "core competencies",
    "key skills",
    "areas of expertise",
    "projects",
    "academic projects",
    "key projects",
    "personal projects",
    "achievements",
    "accomplishments",
    "awards",
    "honors",
    "honors & awards",
    "languages",
    "language proficiency",
    "contact",
    "contact information",
    "contact details",
    "personal details",
    "references",
    *_EDUCATION_ALIASES,
    *_CERTIFICATION_ALIASES,
}

# Degree seniority, highest first -- used to pick `highest_degree` when a
# candidate lists multiple qualifications.
_DEGREE_RANK = [
    "Doctor of Philosophy",
    "Doctor of Medicine",
    "Juris Doctor",
    "Master of Business Administration",
    "Master of Laws",
    "Master of Engineering",
    "Master of Technology",
    "Master of Science",
    "Master of Arts",
    "Master of Computer Applications",
    "Bachelor of Engineering",
    "Bachelor of Technology",
    "Bachelor of Science",
    "Bachelor of Arts",
    "Bachelor of Computer Applications",
    "Bachelor of Laws",
    "Associate Degree",
    "Diploma",
    "High School Diploma",
]
_DEGREE_RANK_INDEX = {name: i for i, name in enumerate(_DEGREE_RANK)}


class EducationCertificationExtractor:
    def __init__(self, output_dir: str | Path = "outputs"):
        self.degree_parser = DegreeParser()
        self.certification_parser = CertificationParser()
        self.store = ResultStore(output_dir)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def extract_from_sections(
        self, sections: Dict[str, str], source_file: str = "unknown"
    ) -> AcademicProfileRecord:
        """
        Extract an academic profile from a pre-segmented section dict.
        Keys are matched case-insensitively against known Education /
        Certifications heading aliases, so callers can pass either
        canonical names ("Education") or raw headings ("Academic Background").
        """
        try:
            education_text = self._find_section(sections, _EDUCATION_ALIASES)
            certification_text = self._find_section(sections, _CERTIFICATION_ALIASES)
            record = self._build_record(education_text, certification_text, source_file)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to extract academic profile for %s", source_file)
            record = self._failed_record(source_file, str(exc))

        self.store.save(record, name_hint=source_file)
        return record

    def extract_from_text(
        self, cleaned_text: str, source_file: str = "unknown"
    ) -> AcademicProfileRecord:
        """
        Extract an academic profile from a full cleaned resume text block
        (e.g. Day 5 ExtractionRecord.cleaned_text) by locating the
        Education / Certifications sections internally first.
        """
        try:
            sections = self._split_into_sections(cleaned_text)
            education_text = self._find_section(sections, _EDUCATION_ALIASES)
            certification_text = self._find_section(sections, _CERTIFICATION_ALIASES)
            record = self._build_record(education_text, certification_text, source_file)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to extract academic profile for %s", source_file)
            record = self._failed_record(source_file, str(exc))

        self.store.save(record, name_hint=source_file)
        return record

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _build_record(
        self, education_text: str, certification_text: str, source_file: str
    ) -> AcademicProfileRecord:
        warnings: List[str] = []

        degrees = self.degree_parser.parse_section(education_text)
        certifications = self.certification_parser.parse_section(certification_text)

        if not education_text.strip():
            warnings.append("No Education section content found.")
        elif not degrees:
            warnings.append("Education section found but no degree could be parsed from it.")

        if not certification_text.strip():
            warnings.append("No Certifications section content found.")

        low_confidence_degrees = [d for d in degrees if d.confidence < 0.5]
        if low_confidence_degrees:
            warnings.append(
                f"{len(low_confidence_degrees)} degree entr"
                f"{'y' if len(low_confidence_degrees) == 1 else 'ies'} parsed with low confidence."
            )

        if degrees or certifications:
            status = "partial" if warnings else "success"
        else:
            status = "failed"
            warnings.append("No degrees or certifications could be extracted.")

        return AcademicProfileRecord(
            source_file=str(source_file),
            schema_version=SCHEMA_VERSION,
            extracted_at=ResultStore.now_iso(),
            degrees=degrees,
            certifications=certifications,
            degrees_found=len(degrees),
            certifications_found=len(certifications),
            highest_degree=self._highest_degree(degrees),
            warnings=warnings,
            status=status,
            error=None,
        )

    def _failed_record(self, source_file: str, error: str) -> AcademicProfileRecord:
        return AcademicProfileRecord(
            source_file=str(source_file),
            schema_version=SCHEMA_VERSION,
            extracted_at=ResultStore.now_iso(),
            degrees=[],
            certifications=[],
            degrees_found=0,
            certifications_found=0,
            highest_degree=None,
            warnings=[],
            status="failed",
            error=error,
        )

    @staticmethod
    def _highest_degree(degrees) -> Optional[str]:
        ranked = [d.degree_type_normalized for d in degrees if d.degree_type_normalized]
        if not ranked:
            return None
        ranked.sort(key=lambda name: _DEGREE_RANK_INDEX.get(name, len(_DEGREE_RANK)))
        return ranked[0]

    @staticmethod
    def _find_section(sections: Dict[str, str], aliases: List[str]) -> str:
        for key, text in sections.items():
            if key.strip().lower() in aliases:
                return text or ""
        return ""

    @staticmethod
    def _split_into_sections(cleaned_text: str) -> Dict[str, str]:
        """
        Fallback section splitter for when only a flat cleaned-text block
        is available (no Day 8 segmentation). Recognizes the same canonical
        heading set used across the project; a line is treated as a heading
        if, on its own, it exactly matches (case-insensitively) one of the
        known section names.
        """
        sections: Dict[str, str] = {}
        current_heading: Optional[str] = None
        buffer: List[str] = []

        def flush():
            if current_heading is not None:
                sections[current_heading] = "\n".join(buffer).strip()

        for line in cleaned_text.splitlines():
            stripped = line.strip()
            heading_key = stripped.strip(":").strip().lower()
            if heading_key in _ALL_KNOWN_HEADINGS and len(stripped) < 40:
                flush()
                current_heading = heading_key
                buffer = []
            elif current_heading is not None:
                buffer.append(line)

        flush()
        return sections
