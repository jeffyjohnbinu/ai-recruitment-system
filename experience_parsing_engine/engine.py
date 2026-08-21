"""
Experience Parsing & Relevance Engine
---------------------------------------
Top-level orchestrator: cleaned resume text -> parser -> gap/overlap
analysis -> (optional) job relevance scoring -> structured storage.

Usage:
    engine = ExperienceParsingEngine(output_dir="outputs")
    record = engine.process_text(resume_text, candidate_id="cand_123", source_file="john.docx")

    relevance = engine.score_relevance(
        record, target_title="Senior Backend Engineer",
        job_keywords=["python", "aws", "microservices"], job_id="job_456",
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from .gaps import analyze_timeline
from .parser import ExperienceParser
from .relevance import ExperienceRelevanceScorer
from .storage import (
    ExperienceRecord,
    ExperienceRelevanceRecord,
    ResultStore,
    build_experience_record,
    build_relevance_record,
)

logger = logging.getLogger("experience_engine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class ExperienceParsingEngine:
    def __init__(self, output_dir: str | Path = "outputs", gap_threshold_months: int = 1):
        self.parser = ExperienceParser()
        self.scorer = ExperienceRelevanceScorer()
        self.store = ResultStore(output_dir)
        self.gap_threshold_months = gap_threshold_months

    def process_text(
        self, resume_text: str, candidate_id: str, source_file: str = ""
    ) -> ExperienceRecord:
        logger.info("Parsing experience for candidate %s", candidate_id)

        try:
            parse_result = self.parser.parse(resume_text)
            timeline = analyze_timeline(parse_result.entries, self.gap_threshold_months)
            record = build_experience_record(
                source_file=source_file or candidate_id,
                candidate_id=candidate_id,
                parse_result=parse_result,
                timeline=timeline,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to parse experience for %s", candidate_id)
            record = ExperienceRecord(
                candidate_id=candidate_id,
                source_file=source_file or candidate_id,
                schema_version="1.0.0",
                extracted_at=self._now(),
                entries=[],
                total_experience_months=0,
                total_experience_years=0.0,
                valid_entry_count=0,
                skipped_entry_count=0,
                gaps=[],
                overlaps=[],
                section_found=False,
                unmatched_line_count=0,
                warnings=[],
                status="failed",
                error=str(exc),
            )

        self.store.save_experience(record)
        return record

    def score_relevance(
        self,
        record: ExperienceRecord,
        target_title: str,
        job_keywords: Iterable[str],
        job_id: str,
    ) -> ExperienceRelevanceRecord:
        """
        Re-parses `record.entries` back into ExperienceEntry objects and
        scores them against a target role. Kept decoupled from
        `process_text` so relevance can be (re)computed against many
        different job postings without re-parsing the resume.
        """
        from .parser import ExperienceEntry

        entries = [ExperienceEntry(**e) for e in record.entries]
        result = self.scorer.score_experience(
            entries=entries,
            target_title=target_title,
            job_keywords=job_keywords,
            candidate_id=record.candidate_id,
            job_id=job_id,
        )
        rel_record = build_relevance_record(record.candidate_id, job_id, result)
        self.store.save_relevance(rel_record)
        return rel_record

    @staticmethod
    def _now() -> str:
        from .storage import now_iso

        return now_iso()
