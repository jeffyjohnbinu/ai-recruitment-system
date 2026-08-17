"""
Job Description Parsing Engine
---------------------------------
Top-level orchestrator: raw JD text/file -> clean -> extract fields ->
build Job Requirement Object -> store.

Usage:
    engine = JDParsingEngine(output_dir="outputs")
    record = engine.process_file("jobs/backend_engineer.txt")
    # or, for text you already have in memory (e.g. pasted from an ATS):
    record = engine.process_text(jd_text, source_name="backend_engineer")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from .cleaner import JDTextCleaner
from .extractors import extract_education, extract_experience, extract_role, extract_skills
from .storage import JDResultStore, JobRequirementRecord

logger = logging.getLogger("jd_parsing_engine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

SUPPORTED_EXTENSIONS = {".txt", ".md"}


class UnsupportedFileTypeError(Exception):
    pass


class JDParsingEngine:
    def __init__(self, output_dir: str | Path = "outputs"):
        self.cleaner = JDTextCleaner()
        self.store = JDResultStore(output_dir)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def process_file(self, path: str | Path) -> JobRequirementRecord:
        path = Path(path)
        ext = path.suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}. "
                "For PDF/DOCX job postings, extract text first with "
                "resume_extraction_engine's readers, then pass the text to process_text()."
            )

        if not path.exists():
            raise FileNotFoundError(f"Job description file not found: {path}")

        logger.info("Processing %s", path.name)
        raw_text = path.read_text(encoding="utf-8", errors="replace")

        try:
            record = self._build_record(source_file=str(path), raw_text=raw_text)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to process %s", path)
            record = self._failed_record(str(path), str(exc))

        self.store.save(record)
        return record

    def process_text(self, text: str, source_name: str = "job_description") -> JobRequirementRecord:
        """Parse JD text directly (no file on disk required)."""
        logger.info("Processing in-memory JD text: %s", source_name)
        try:
            record = self._build_record(source_file=source_name, raw_text=text)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to process JD text %s", source_name)
            record = self._failed_record(source_name, str(exc))

        self.store.save(record)
        return record

    def process_directory(self, directory: str | Path) -> List[JobRequirementRecord]:
        directory = Path(directory)
        records = []
        for file in sorted(directory.iterdir()):
            if file.suffix.lower() in SUPPORTED_EXTENSIONS:
                records.append(self.process_file(file))
        return records

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _build_record(self, source_file: str, raw_text: str) -> JobRequirementRecord:
        cleaned_text, report = self.cleaner.clean(raw_text)
        sections_detected = self.cleaner.detect_sections(cleaned_text)

        requirements_text = self.cleaner.section_text(cleaned_text, "Requirements")
        skills_text = self.cleaner.section_text(cleaned_text, "Skills")
        preferred_text = self.cleaner.section_text(cleaned_text, "Preferred Qualifications")
        education_text = self.cleaner.section_text(cleaned_text, "Education")

        combined_required_scope = "\n".join([requirements_text, skills_text]).strip()
        combined_education_scope = "\n".join([education_text, requirements_text]).strip()

        role_info = extract_role(cleaned_text)
        skills_info = extract_skills(cleaned_text, combined_required_scope, preferred_text)
        experience_info = extract_experience(cleaned_text)
        education_info = extract_education(cleaned_text, combined_education_scope)

        warnings: List[str] = []
        if not cleaned_text.strip():
            warnings.append("No text could be extracted from this job description.")
        if not skills_info.required_skills and not skills_info.preferred_skills:
            warnings.append(
                "No recognized skills were detected — consider extending the skill dictionary."
            )
        if role_info.normalized_role is None:
            warnings.append("Role title could not be normalized against the known role list.")
        if experience_info.min_years is None:
            warnings.append("No explicit years-of-experience requirement was detected.")
        if education_info.degree_level is None:
            warnings.append("No explicit education/degree requirement was detected.")

        status = "success"
        if not cleaned_text.strip():
            status = "failed"
        elif warnings:
            status = "partial"

        return JobRequirementRecord(
            source_file=source_file,
            extracted_at=JDResultStore.now_iso(),
            raw_title=role_info.raw_title,
            normalized_role=role_info.normalized_role,
            seniority_level=role_info.seniority_level,
            required_skills=skills_info.required_skills,
            preferred_skills=skills_info.preferred_skills,
            min_experience_years=experience_info.min_years,
            max_experience_years=experience_info.max_years,
            experience_mentions=experience_info.raw_mentions,
            education_level=education_info.degree_level,
            education_fields=education_info.fields_of_study,
            education_mentions=education_info.raw_mentions,
            raw_char_count=len(raw_text),
            cleaned_char_count=len(cleaned_text),
            sections_detected=sections_detected,
            warnings=warnings,
            cleaning_stats={
                "removed_control_chars": report.removed_control_chars,
                "normalized_bullets": report.normalized_bullets,
                "normalized_headings": report.normalized_headings,
                "dehyphenated_words": report.dehyphenated_words,
                "dropped_noise_lines": report.dropped_noise_lines,
            },
            cleaned_text=cleaned_text,
            status=status,
        )

    @staticmethod
    def _failed_record(source_file: str, error: str) -> JobRequirementRecord:
        return JobRequirementRecord(
            source_file=source_file,
            extracted_at=JDResultStore.now_iso(),
            raw_title=None,
            normalized_role=None,
            seniority_level=None,
            required_skills=[],
            preferred_skills=[],
            min_experience_years=None,
            max_experience_years=None,
            experience_mentions=[],
            education_level=None,
            education_fields=[],
            education_mentions=[],
            raw_char_count=0,
            cleaned_char_count=0,
            sections_detected=[],
            warnings=[],
            cleaning_stats={},
            cleaned_text="",
            status="failed",
            error=error,
        )
