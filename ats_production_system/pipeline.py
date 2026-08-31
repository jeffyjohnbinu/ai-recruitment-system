"""
pipeline.py
------------
Day 20 deliverable — Zecpath AI Job Portal

ATSPipeline: the single, real, end-to-end orchestrator for the ATS AI
module. Wires together every prior day's engine in the actual sequence
a production ATS needs:

    resume file ─┐
                  ├─> Day 5  ResumeExtractionEngine   (extract + clean)
                  ├─> Day 18 PerformanceOptimizationEngine.repair_noisy_text (best-effort)
                  ├─> Day 8  SectionClassifierEngine   (segment into sections)
                  ├─> Day 9  SkillExtractionEngine      (candidate skills)
                  ├─> Day 10 ExperienceParsingEngine    (timeline + relevance)
                  └─> Day 11 EducationCertificationExtractor (degrees/certs)

    job file ────> Day 6  JDParsingEngine (required/preferred skills,
                            experience & education requirements)

    (resume outputs, job output) ─> Day 12 SemanticMatchingEngine
                                  ─> component_adapters (Day 20 field-mapping fix)
                                  ─> Day 13 ATSScoringEngine (explainable final score)
                                  ─> Day 14 CandidateRankingEngine (per-job ranking, batch mode)
                                  ─> Day 15 FairnessBiasEngine (pool-level normalization + audit)
                                  ─> Day 20 PipelineResultStore (Day 7 envelope)

Additive-only: this module imports Day 5-18 packages as-is and never
modifies them. Every stage is wrapped so a missing/failing upstream
package degrades that one component to "unavailable" (logged as a
warning) rather than crashing the whole run -- consistent with the
project's two-pass fallback discipline. Unlike the Day 17 test harness's
`pipeline_adapter.py`, the import paths and output-field mappings below
were verified against the real Day 5-18 source, not guessed at (see
component_adapters.py for the specific field-mapping fixes this
required).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import component_adapters as adapters
from .storage import PipelineResultStore, PipelineRunRecord

logger = logging.getLogger("ats_production_system.pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class ATSPipeline:
    def __init__(self, output_dir: str | Path = "outputs", embedder_preference: str = "auto"):
        self.output_dir = Path(output_dir)
        self.store = PipelineResultStore(self.output_dir)
        self.embedder_preference = embedder_preference
        self._engines: Dict[str, Any] = {}
        self._availability: Dict[str, bool] = {}
        self._init_engines()

    # ------------------------------------------------------------------ #
    # Engine bootstrap -- every engine is optional; missing ones degrade
    # the relevant pipeline stage instead of raising at import time.
    # ------------------------------------------------------------------ #
    def _init_engines(self) -> None:
        engine_dir = str(self.output_dir)

        def _try(name: str, factory):
            try:
                self._engines[name] = factory()
                self._availability[name] = True
            except Exception as exc:  # noqa: BLE001
                logger.warning("Engine unavailable: %s (%s)", name, exc)
                self._engines[name] = None
                self._availability[name] = False

        def _resume_extraction():
            from resume_extraction_engine import ResumeExtractionEngine

            return ResumeExtractionEngine(output_dir=engine_dir)

        def _jd_parsing():
            from jd_parsing_engine import JDParsingEngine

            return JDParsingEngine(output_dir=engine_dir)

        def _section_classifier():
            from resume_section_classifier import SectionClassifierEngine

            return SectionClassifierEngine(output_dir=engine_dir)

        def _skill_extraction():
            from skill_extraction_engine import SkillExtractionEngine

            return SkillExtractionEngine(output_dir=engine_dir)

        def _experience_parsing():
            from experience_parsing_engine import ExperienceParsingEngine

            return ExperienceParsingEngine(output_dir=engine_dir)

        def _education_extraction():
            from education_certification_extractor import EducationCertificationExtractor

            return EducationCertificationExtractor(output_dir=engine_dir)

        def _semantic_matching():
            from semantic_matching_engine import SemanticMatchingEngine

            return SemanticMatchingEngine(
                output_dir=engine_dir, embedder_preference=self.embedder_preference
            )

        def _ats_scoring():
            from ats_scoring_engine import ATSScoringEngine

            return ATSScoringEngine()

        def _candidate_ranking():
            from candidate_ranking_engine import CandidateRankingEngine

            return CandidateRankingEngine(output_dir=engine_dir)

        def _fairness_bias():
            from fairness_bias_engine import FairnessBiasEngine

            return FairnessBiasEngine()

        def _performance_optimizer():
            from performance_optimization_engine import PerformanceOptimizationEngine

            return PerformanceOptimizationEngine()

        _try("resume_extraction", _resume_extraction)
        _try("jd_parsing", _jd_parsing)
        _try("section_classifier", _section_classifier)
        _try("skill_extraction", _skill_extraction)
        _try("experience_parsing", _experience_parsing)
        _try("education_extraction", _education_extraction)
        _try("semantic_matching", _semantic_matching)
        _try("ats_scoring", _ats_scoring)
        _try("candidate_ranking", _candidate_ranking)
        _try("fairness_bias", _fairness_bias)
        _try("performance_optimizer", _performance_optimizer)

        available = [k for k, v in self._availability.items() if v]
        unavailable = [k for k, v in self._availability.items() if not v]
        logger.info("ATSPipeline ready. Available engines: %s", available)
        if unavailable:
            logger.warning("Degraded (unavailable) engines: %s", unavailable)

    def engine_status(self) -> Dict[str, bool]:
        return dict(self._availability)

    # ------------------------------------------------------------------ #
    # Job-side processing (parsed once per job, reused across candidates)
    # ------------------------------------------------------------------ #
    def process_job(self, job_path: str | Path) -> Dict[str, Any]:
        engine = self._engines.get("jd_parsing")
        if engine is None:
            raise RuntimeError("jd_parsing_engine is not available; cannot process job postings.")
        record = engine.process_file(job_path)
        return record.to_dict()

    # ------------------------------------------------------------------ #
    # Candidate-side processing + scoring against one job
    # ------------------------------------------------------------------ #
    def process_candidate(
        self,
        resume_path: str | Path,
        job_record: Dict[str, Any],
        candidate_id: str,
        job_id: str,
        role: Optional[str] = None,
    ) -> PipelineRunRecord:
        resume_path = Path(resume_path)
        stages_completed: List[str] = []
        stages_failed: List[str] = []
        engines_used: Dict[str, str] = {}
        warnings: List[str] = []

        # --- Stage 1: extraction (Day 5) ---
        extraction = self._engines.get("resume_extraction")
        if extraction is None:
            return self._failed_run(
                candidate_id,
                job_id,
                resume_path,
                job_record,
                "resume_extraction_engine unavailable",
            )
        try:
            ext_record = extraction.process_file(resume_path)
            cleaned_text = ext_record.cleaned_text
            stages_completed.append("resume_extraction")
            engines_used["resume_extraction"] = ext_record.engine_used
            if ext_record.status != "success":
                warnings.append(f"Resume extraction status={ext_record.status}")
        except Exception as exc:  # noqa: BLE001
            return self._failed_run(candidate_id, job_id, resume_path, job_record, str(exc))

        # --- Stage 2: noise repair (Day 18, best-effort) ---
        optimizer = self._engines.get("performance_optimizer")
        if optimizer is not None:
            try:
                repaired_text, _report = optimizer.repair_noisy_text(cleaned_text)
                if repaired_text:
                    cleaned_text = repaired_text
                stages_completed.append("performance_optimizer.repair_noisy_text")
            except Exception as exc:  # noqa: BLE001
                stages_failed.append("performance_optimizer.repair_noisy_text")
                warnings.append(f"Noise repair skipped: {exc}")
        else:
            stages_failed.append("performance_optimizer")

        # --- Stage 3: section classification (Day 8) ---
        section_map: Dict[str, str] = {}
        classifier = self._engines.get("section_classifier")
        if classifier is not None:
            try:
                class_record = classifier.classify_text(cleaned_text, source_name=resume_path.name)
                for block in class_record.blocks:
                    section_map.setdefault(block.label, "")
                    section_map[block.label] += (
                        "\n" + block.text if section_map[block.label] else block.text
                    )
                stages_completed.append("section_classifier")
            except Exception as exc:  # noqa: BLE001
                stages_failed.append("section_classifier")
                warnings.append(f"Section classification failed: {exc}")
        else:
            stages_failed.append("section_classifier")

        # --- Stage 4: skill extraction (Day 9) ---
        candidate_skill_names: List[str] = []
        skill_extraction = self._engines.get("skill_extraction")
        if skill_extraction is not None:
            try:
                skill_result = skill_extraction.extract_from_text(
                    cleaned_text, source_name=resume_path.name, sections=list(section_map.keys())
                )
                candidate_skill_names = [s.skill for s in skill_result.skills]
                stages_completed.append("skill_extraction")
            except Exception as exc:  # noqa: BLE001
                stages_failed.append("skill_extraction")
                warnings.append(f"Skill extraction failed: {exc}")
        else:
            stages_failed.append("skill_extraction")

        # --- Stage 5: experience parsing + relevance (Day 10) ---
        experience_component: Dict[str, Any] = {}
        experience_engine = self._engines.get("experience_parsing")
        if experience_engine is not None:
            try:
                exp_record = experience_engine.process_text(
                    cleaned_text, candidate_id=candidate_id, source_file=str(resume_path)
                )
                job_keywords = (job_record.get("required_skills") or []) + (
                    job_record.get("preferred_skills") or []
                )
                rel_record = experience_engine.score_relevance(
                    exp_record,
                    target_title=job_record.get("normalized_role")
                    or job_record.get("raw_title")
                    or "",
                    job_keywords=job_keywords,
                    job_id=job_id,
                )
                experience_component = adapters.build_experience_component(rel_record.to_dict())
                stages_completed.append("experience_parsing")
            except Exception as exc:  # noqa: BLE001
                stages_failed.append("experience_parsing")
                warnings.append(f"Experience parsing/relevance failed: {exc}")
        else:
            stages_failed.append("experience_parsing")

        # --- Stage 6: education/certification extraction (Day 11) ---
        education_component: Dict[str, Any] = {}
        education_engine = self._engines.get("education_extraction")
        if education_engine is not None:
            try:
                if section_map:
                    edu_record = education_engine.extract_from_sections(
                        section_map, source_file=str(resume_path)
                    )
                else:
                    edu_record = education_engine.extract_from_text(
                        cleaned_text, source_file=str(resume_path)
                    )
                education_component = adapters.build_education_component(
                    edu_record.to_dict(),
                    job_record.get("education_level"),
                    job_record.get("education_fields"),
                )
                stages_completed.append("education_extraction")
            except Exception as exc:  # noqa: BLE001
                stages_failed.append("education_extraction")
                warnings.append(f"Education extraction failed: {exc}")
        else:
            stages_failed.append("education_extraction")

        # --- Stage 7: semantic matching (Day 12) ---
        semantic_component: Dict[str, Any] = {}
        semantic_engine = self._engines.get("semantic_matching")
        semantic_record_dict: Dict[str, Any] = {}
        if semantic_engine is not None:
            try:
                resume_input = section_map if section_map else cleaned_text
                match_record = semantic_engine.match(
                    candidate_id=candidate_id,
                    job_id=job_id,
                    resume_input=resume_input,
                    job_input=job_record,
                )
                semantic_record_dict = match_record.to_dict()
                semantic_component = adapters.build_semantic_component(semantic_record_dict)
                engines_used["semantic_matching"] = match_record.engine_used
                stages_completed.append("semantic_matching")
            except Exception as exc:  # noqa: BLE001
                stages_failed.append("semantic_matching")
                warnings.append(f"Semantic matching failed: {exc}")
        else:
            stages_failed.append("semantic_matching")

        # --- Stage 8: skill component mapping (Day 20 adapter) ---
        skill_component = adapters.build_skill_component(
            candidate_skill_names,
            job_record.get("required_skills") or [],
            job_record.get("preferred_skills") or [],
        )

        # --- Stage 9: ATS scoring (Day 13) ---
        scoring_engine = self._engines.get("ats_scoring")
        if scoring_engine is None:
            return self._failed_run(
                candidate_id, job_id, resume_path, job_record, "ats_scoring_engine unavailable"
            )
        try:
            scoring_result = scoring_engine.compute_score(
                candidate_id=candidate_id,
                job_id=job_id,
                skill_data=skill_component,
                experience_data=experience_component or None,
                education_data=education_component or None,
                semantic_data=semantic_component or None,
                role=role,
            )
            stages_completed.append("ats_scoring")
        except Exception as exc:  # noqa: BLE001
            return self._failed_run(candidate_id, job_id, resume_path, job_record, str(exc))

        from ats_scoring_engine.explainability import build_narrative

        narrative = build_narrative(scoring_result)
        component_dicts = [
            {
                "name": c.name,
                "available": c.available,
                "raw_score": c.raw_score,
                "weight_original": c.weight_original,
                "weight_applied": c.weight_applied,
                "weighted_contribution": c.weighted_contribution,
                "notes": c.notes,
            }
            for c in scoring_result.components
        ]

        run_status = (
            "success"
            if scoring_result.status == "scored" and not stages_failed
            else ("partial" if scoring_result.status == "scored" else "failed")
        )

        candidate_profile = {
            "candidate_id": candidate_id,
            "source_file": str(resume_path),
            "skills": candidate_skill_names,
            "total_experience_years": experience_component.get("total_experience_years"),
            "highest_degree": education_component.get("highest_degree"),
            "certifications": education_component.get("relevant_certifications", []),
        }

        record = PipelineRunRecord(
            schema_version="1.0.0",
            model_version="ats-production-system-1.0.0",
            pipeline_version="zecpath-day20",
            candidate_id=candidate_id,
            job_id=job_id,
            generated_at=self.store.now_iso(),
            request_id=self.store.new_request_id(),
            resume_source_file=str(resume_path),
            job_source_file=str(job_record.get("source_file", "")),
            stages_completed=stages_completed,
            stages_failed=stages_failed,
            engines_used=engines_used,
            final_score=scoring_result.final_score,
            match_band=semantic_record_dict.get("match_band"),
            recommendation=None,
            zone=None,
            component_breakdown=component_dicts,
            narrative_explanation=narrative,
            warnings=warnings + scoring_result.warnings,
            status=run_status,
            candidate_profile=candidate_profile,
        )
        self.store.save_run(record)
        return record

    def _failed_run(
        self,
        candidate_id: str,
        job_id: str,
        resume_path: Path,
        job_record: Dict[str, Any],
        error: str,
    ) -> PipelineRunRecord:
        record = PipelineRunRecord(
            schema_version="1.0.0",
            model_version="ats-production-system-1.0.0",
            pipeline_version="zecpath-day20",
            candidate_id=candidate_id,
            job_id=job_id,
            generated_at=self.store.now_iso(),
            request_id=self.store.new_request_id(),
            resume_source_file=str(resume_path),
            job_source_file=str(job_record.get("source_file", "")),
            stages_completed=[],
            stages_failed=["pipeline"],
            engines_used={},
            final_score=None,
            match_band=None,
            recommendation=None,
            zone=None,
            component_breakdown=[],
            narrative_explanation=None,
            warnings=[],
            status="failed",
            error=error,
        )
        self.store.save_run(record)
        return record

    # ------------------------------------------------------------------ #
    # Batch mode: many candidates against one job -> ranked + audited pool
    # ------------------------------------------------------------------ #
    def run_batch(
        self,
        resume_paths: List[str | Path],
        job_path: str | Path,
        job_id: str,
        role: Optional[str] = None,
    ) -> Dict[str, Any]:
        job_record = self.process_job(job_path)

        runs: List[PipelineRunRecord] = []
        for resume_path in resume_paths:
            candidate_id = Path(resume_path).stem
            run = self.process_candidate(
                resume_path, job_record, candidate_id=candidate_id, job_id=job_id, role=role
            )
            runs.append(run)

        # --- Day 14: rank the scored candidates ---
        ranking_result = None
        ranking_engine = self._engines.get("candidate_ranking")
        if ranking_engine is not None:
            try:
                from candidate_ranking_engine.storage import CandidateMatchInput

                match_inputs = [
                    CandidateMatchInput(
                        candidate_id=r.candidate_id,
                        job_id=r.job_id,
                        overall_score=r.final_score if r.final_score is not None else 0.0,
                        band=r.match_band or "unscored",
                        section_scores={
                            c["name"]: c["raw_score"]
                            for c in r.component_breakdown
                            if c["raw_score"] is not None
                        },
                        source_file=r.resume_source_file,
                    )
                    for r in runs
                    if r.status != "failed"
                ]
                if match_inputs:
                    ranking_result = ranking_engine.rank_job(match_inputs, job_id=job_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Candidate ranking failed: %s", exc)

        # --- Day 15: pool-level fairness + bias audit ---
        fairness_result = None
        fairness_engine = self._engines.get("fairness_bias")
        if fairness_engine is not None:
            try:
                raw_profiles = [
                    r.candidate_profile
                    or {"candidate_id": r.candidate_id, "source_file": r.resume_source_file}
                    for r in runs
                    if r.status != "failed"
                ]
                score_components = {
                    r.candidate_id: {
                        c["name"]: (c["raw_score"] or 0.0, c["weight_applied"])
                        for c in r.component_breakdown
                        if c["available"]
                    }
                    for r in runs
                    if r.status != "failed"
                }
                if raw_profiles:
                    fairness_result = fairness_engine.process_candidate_pool(
                        raw_profiles=raw_profiles,
                        score_components=score_components,
                        role_id=role,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Fairness/bias processing failed: %s", exc)

        zones_by_candidate = {}
        if ranking_result is not None:
            for ranked in ranking_result.ranked_candidates:
                zones_by_candidate[ranked.candidate_id] = ranked.zone

        candidates_summary = []
        for r in runs:
            candidates_summary.append(
                {
                    "candidate_id": r.candidate_id,
                    "final_score": r.final_score,
                    "match_band": r.match_band,
                    "zone": zones_by_candidate.get(r.candidate_id),
                    "status": r.status,
                    "stages_failed": r.stages_failed,
                }
            )

        from .storage import BatchRunSummary

        shortlist_count = sum(1 for c in candidates_summary if c["zone"] == "Shortlist")
        review_count = sum(1 for c in candidates_summary if c["zone"] == "Review")
        reject_count = sum(1 for c in candidates_summary if c["zone"] == "Auto-Reject")

        summary = BatchRunSummary(
            schema_version="1.0.0",
            generated_at=self.store.now_iso(),
            job_id=job_id,
            total_candidates=len(runs),
            shortlist_count=shortlist_count,
            review_count=review_count,
            auto_reject_count=reject_count,
            candidates=candidates_summary,
        )
        self.store.save_batch(summary)

        fairness_record = None
        if fairness_result is not None:
            try:
                fairness_record = fairness_result.to_storage_record()
                fairness_dir = self.output_dir / "structured" / "fairness_reports"
                fairness_dir.mkdir(parents=True, exist_ok=True)
                import json as _json

                with open(fairness_dir / f"{job_id}.json", "w", encoding="utf-8") as f:
                    _json.dump(
                        fairness_record.to_dict(), f, indent=2, ensure_ascii=False, default=str
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to persist fairness report: %s", exc)

        return {
            "runs": runs,
            "ranking": ranking_result,
            "fairness": fairness_result,
            "summary": summary,
        }
