"""
Skill Extraction Engine
-------------------------
Day 9 deliverable — Zecpath AI Job Portal

Extracts technical, business, and creative skills from cleaned resume
text (Day 5's `cleaned_text` output, optionally alongside Day 8's
section boundaries), with synonym/spelling-variant handling, skill-stack
expansion, and explainable confidence scoring.

Usage:
    engine = SkillExtractionEngine(output_dir="outputs")
    result = engine.extract_from_text(cleaned_text, source_name="john_doe.docx")

    # With Day 8 section labels for a section-aware confidence bonus:
    result = engine.extract_from_text(
        cleaned_text, source_name="john_doe.docx", sections=["Skills", "Experience", ...]
    )
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .matcher import SkillMatcher
from .normalizer import normalize_mentions
from .storage import SkillExtractionResult, SkillResultStore

# Canonical section-heading lines as produced by Day 5's TextCleaner
# (resume_extraction_engine/cleaner.py::_SECTION_ALIASES). Kept as a local
# copy rather than a cross-package import so this module has no hard
# dependency on Day 5's package layout.
_CANONICAL_HEADINGS = {
    "Summary",
    "Experience",
    "Education",
    "Skills",
    "Projects",
    "Certifications",
    "Achievements",
    "Languages",
    "Contact",
    "References",
}

_HEADING_LINE_RE = re.compile(
    r"^(" + "|".join(re.escape(h) for h in _CANONICAL_HEADINGS) + r")\s*$",
    re.MULTILINE,
)


class SkillExtractionEngine:
    def __init__(self, output_dir: str | Path = "outputs"):
        self.matcher = SkillMatcher()
        self.store = SkillResultStore(output_dir)

    # ------------------------------------------------------------------ #
    def extract_from_text(
        self,
        text: str,
        source_name: str = "input",
        sections: Optional[List[str]] = None,
        persist: bool = True,
    ) -> SkillExtractionResult:
        """
        Run the full extraction pipeline over `text`.

        `sections` is accepted for interface parity with Day 8's
        classifier output but is currently unused directly -- section
        boundaries are instead detected from the canonical heading lines
        already present in Day 5's cleaned_text (see
        `_detect_section_spans`). Passing `sections` lets a caller assert
        which sections *should* be present without changing behavior.
        """
        warnings: List[str] = []

        if not text or not text.strip():
            result = SkillExtractionResult(
                source_file=source_name,
                extracted_at=SkillResultStore.now_iso(),
                input_char_count=0,
                total_skills_found=0,
                skills_by_category={},
                skills=[],
                status="failed",
                warnings=["No input text provided."],
                error="empty_input",
            )
            if persist:
                self.store.save(result, name_hint=source_name)
            return result

        section_spans = self._detect_section_spans(text)

        phrase_mentions, covered_spans = self.matcher.match_phrases(text)
        stack_mentions = self.matcher.match_stacks(text)
        fuzzy_mentions = self.matcher.match_fuzzy(text, covered_spans)

        all_mentions = phrase_mentions + stack_mentions + fuzzy_mentions

        span_sections = {
            m.start: self._section_for_offset(section_spans, m.start) for m in all_mentions
        }
        span_sections = {k: v for k, v in span_sections.items() if v is not None}

        records = normalize_mentions(all_mentions, span_sections=span_sections)

        skills_by_category: Dict[str, int] = {}
        for record in records:
            skills_by_category[record.category] = skills_by_category.get(record.category, 0) + 1

        if not records:
            warnings.append("No skills matched against the master dictionary.")

        status = "success" if records else "partial"

        result = SkillExtractionResult(
            source_file=source_name,
            extracted_at=SkillResultStore.now_iso(),
            input_char_count=len(text),
            total_skills_found=len(records),
            skills_by_category=skills_by_category,
            skills=records,
            status=status,
            warnings=warnings,
        )

        if persist:
            self.store.save(result, name_hint=source_name)

        return result

    def extract_from_file(
        self, path: str | Path, sections: Optional[List[str]] = None
    ) -> SkillExtractionResult:
        """Convenience wrapper: read a plain-text/cleaned-text file and extract."""
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        return self.extract_from_text(text, source_name=path.name, sections=sections)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _detect_section_spans(text: str) -> List[Tuple[str, int, int]]:
        """
        Return [(section_name, start_offset, end_offset), ...] by locating
        canonical heading lines and treating the text up to the next
        heading (or end of text) as that section's span.
        """
        headings = list(_HEADING_LINE_RE.finditer(text))
        if not headings:
            return []

        spans: List[Tuple[str, int, int]] = []
        for i, match in enumerate(headings):
            name = match.group(1)
            start = match.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            spans.append((name, start, end))
        return spans

    @staticmethod
    def _section_for_offset(spans: List[Tuple[str, int, int]], offset: int) -> Optional[str]:
        for name, start, end in spans:
            if start <= offset < end:
                return name
        return None
