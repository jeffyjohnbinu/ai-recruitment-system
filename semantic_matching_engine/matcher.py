"""
Semantic Matching Engine
-------------------------
Top-level orchestrator: (resume data, job data) -> section extraction ->
embeddings -> similarity -> threshold classification -> SemanticMatchRecord.

Cross-day integration
----------------------
Per the project's integration discipline (each new module accepts prior
days' outputs directly), `SemanticMatchingEngine.match()` accepts resume
and job inputs in any of these shapes and normalizes them internally via
`adapters.py`:

  - A plain dict already shaped like {"skills": str, "experience": str,
    "projects": str} -- the simplest path, always supported.
  - A Day 8 `resume_section_classifier` style section map
    ({"Skills": "...", "Experience": "...", ...}).
  - A Day 9 `skill_extraction_engine` style result (list of extracted
    skill dicts, each with at least a "skill"/"name" key).
  - A Day 10 `experience_parsing_engine` style result (list of experience
    entry dicts with descriptive text fields).
  - A Day 6 `JobRequirementRecord`-style dict (role, skills, experience,
    education) for the job description side.
  - Raw flat text (Day 5 `ResumeExtractionEngine` cleaned_text) as a last
    resort -- used for every section, which still lets the engine run
    (skills/experience/projects sections just aren't isolated from each
    other, which is reflected via a warning on the record).

Usage:
    engine = SemanticMatchingEngine(output_dir="outputs")
    record = engine.match(candidate_id="C001", job_id="J001",
                           resume_input=..., job_input=...)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .adapters import normalize_job_input, normalize_resume_input
from .embeddings import Embedder, get_embedder
from .similarity import DEFAULT_SECTION_WEIGHTS, compute_similarity
from .storage import SCHEMA_VERSION, ResultStore, SemanticMatchRecord
from .thresholds import ThresholdConfig

logger = logging.getLogger("semantic_matching_engine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class SemanticMatchingEngine:
    def __init__(
        self,
        output_dir: str | Path = "outputs",
        embedder: Optional[Embedder] = None,
        thresholds: Optional[ThresholdConfig] = None,
        section_weights: Optional[Dict[str, float]] = None,
        embedder_preference: str = "auto",
    ):
        self.embedder = embedder or get_embedder(prefer=embedder_preference)
        self.thresholds = thresholds or ThresholdConfig()
        self.section_weights = section_weights or dict(DEFAULT_SECTION_WEIGHTS)
        self.store = ResultStore(output_dir)
        logger.info(
            "Semantic Matching Engine ready (embedding engine: %s)", self.embedder.engine_name
        )

    def match(
        self,
        candidate_id: str,
        job_id: str,
        resume_input: Any,
        job_input: Any,
        persist: bool = True,
    ) -> SemanticMatchRecord:
        warnings: list = []

        try:
            resume_sections, resume_warnings = normalize_resume_input(resume_input)
            job_sections, job_warnings = normalize_job_input(job_input)
            warnings.extend(resume_warnings)
            warnings.extend(job_warnings)

            breakdown = compute_similarity(
                resume_sections=resume_sections,
                job_sections=job_sections,
                embedder=self.embedder,
                weights=self.section_weights,
            )

            match_band = self.thresholds.classify(breakdown.overall_score)
            is_match = self.thresholds.is_match(breakdown.overall_score)

            status = "success"
            if all(cmp.empty for cmp in breakdown.section_scores.values()):
                status = "failed"
                warnings.append("No comparable text found on either the resume or job side.")
            elif warnings:
                status = "partial"

            record = SemanticMatchRecord(
                candidate_id=candidate_id,
                job_id=job_id,
                schema_version=SCHEMA_VERSION,
                matched_at=ResultStore.now_iso(),
                engine_used=self.embedder.engine_name,
                overall_score=breakdown.overall_score,
                match_band=match_band,
                is_match=is_match,
                section_scores=breakdown.as_dict()["sections"],
                thresholds=self.thresholds.to_dict(),
                warnings=warnings,
                status=status,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to match candidate=%s job=%s", candidate_id, job_id)
            record = SemanticMatchRecord(
                candidate_id=candidate_id,
                job_id=job_id,
                schema_version=SCHEMA_VERSION,
                matched_at=ResultStore.now_iso(),
                engine_used=getattr(self.embedder, "engine_name", "unknown"),
                overall_score=0.0,
                match_band="no_match",
                is_match=False,
                section_scores={},
                thresholds=self.thresholds.to_dict(),
                warnings=warnings,
                status="failed",
                error=str(exc),
            )

        if persist:
            self.store.save(record)

        return record

    def match_many(
        self,
        pairs: list,
        persist: bool = True,
    ) -> list:
        """
        Batch-match a list of (candidate_id, job_id, resume_input, job_input)
        tuples. Useful for validating accuracy across multiple job types
        (Day 12 deliverable: "Validate across multiple job types").
        """
        results = []
        for candidate_id, job_id, resume_input, job_input in pairs:
            results.append(
                self.match(candidate_id, job_id, resume_input, job_input, persist=persist)
            )
        return results
