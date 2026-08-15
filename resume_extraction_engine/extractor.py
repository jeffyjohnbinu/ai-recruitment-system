"""
Resume Extraction Engine
-------------------------
Top-level orchestrator: file -> reader -> cleaner -> normalizer -> storage.

Usage:
    engine = ResumeExtractionEngine(output_dir="outputs")
    record = engine.process_file("resumes/john_doe.pdf")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from .cleaner import _SECTION_ALIASES, TextCleaner, TextNormalizer
from .readers.docx_reader import DOCXReader
from .readers.pdf_reader import PDFReader
from .storage import ExtractionRecord, ResultStore

logger = logging.getLogger("resume_engine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


class UnsupportedFileTypeError(Exception):
    pass


class ResumeExtractionEngine:
    def __init__(self, output_dir: str | Path = "outputs", ocr_fallback: bool = True):
        self.pdf_reader = PDFReader(ocr_fallback=ocr_fallback)
        self.docx_reader = DOCXReader()
        self.cleaner = TextCleaner()
        self.normalizer = TextNormalizer()
        self.store = ResultStore(output_dir)

    def process_file(self, path: str | Path) -> ExtractionRecord:
        path = Path(path)
        ext = path.suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        if not path.exists():
            raise FileNotFoundError(f"Resume file not found: {path}")

        logger.info("Processing %s", path.name)

        try:
            if ext == ".pdf":
                record = self._process_pdf(path)
            else:
                record = self._process_docx(path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to process %s", path)
            record = ExtractionRecord(
                source_file=str(path),
                file_type=ext.lstrip("."),
                engine_used="none",
                extracted_at=ResultStore.now_iso(),
                raw_char_count=0,
                cleaned_char_count=0,
                sections_detected=[],
                tables_found=0,
                images_found=0,
                warnings=[],
                cleaning_stats={},
                cleaned_text="",
                status="failed",
                error=str(exc),
            )

        self.store.save(record)
        return record

    def process_directory(self, directory: str | Path) -> List[ExtractionRecord]:
        directory = Path(directory)
        records = []
        for file in sorted(directory.iterdir()):
            if file.suffix.lower() in SUPPORTED_EXTENSIONS:
                records.append(self.process_file(file))
        return records

    # ------------------------------------------------------------------ #
    def _process_pdf(self, path: Path) -> ExtractionRecord:
        result = self.pdf_reader.read(path)
        raw_text = result.raw_text
        return self._build_record(
            path=path,
            file_type="pdf",
            engine_used=result.engine_used,
            raw_text=raw_text,
            tables_found=sum(len(p.tables) for p in result.pages),
            images_found=sum(1 for p in result.pages if p.had_images),
            warnings=result.warnings,
        )

    def _process_docx(self, path: Path) -> ExtractionRecord:
        result = self.docx_reader.read(path)
        raw_text = result.raw_text
        return self._build_record(
            path=path,
            file_type="docx",
            engine_used=result.engine_used,
            raw_text=raw_text,
            tables_found=len(result.tables),
            images_found=result.image_count,
            warnings=result.warnings,
        )

    def _build_record(
        self,
        path: Path,
        file_type: str,
        engine_used: str,
        raw_text: str,
        tables_found: int,
        images_found: int,
        warnings: List[str],
    ) -> ExtractionRecord:
        cleaned_text, report = self.cleaner.clean(raw_text)
        cleaned_text = self.normalizer.normalize_capitalization(cleaned_text)

        sections_detected = self._detect_sections(cleaned_text)

        status = "success"
        if not cleaned_text.strip():
            status = "failed"
            warnings = warnings + ["No text could be extracted from this file."]
        elif warnings:
            status = "partial"

        return ExtractionRecord(
            source_file=str(path),
            file_type=file_type,
            engine_used=engine_used,
            extracted_at=ResultStore.now_iso(),
            raw_char_count=len(raw_text),
            cleaned_char_count=len(cleaned_text),
            sections_detected=sections_detected,
            tables_found=tables_found,
            images_found=images_found,
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
    def _detect_sections(cleaned_text: str) -> List[str]:
        canonical_headings = set(_SECTION_ALIASES.keys())
        found = []
        for line in cleaned_text.splitlines():
            stripped = line.strip()
            if stripped in canonical_headings and stripped not in found:
                found.append(stripped)
        return found
