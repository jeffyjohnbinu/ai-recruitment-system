"""
scoring_engine.py
-----------------
Day 26 deliverable — Zecpath AI Job Portal

Main scoring engine that orchestrates per-question scoring and session
aggregation.

Public API:
    engine = ScreeningScoringEngine()
    result = engine.score_session(candidate_id, job_id, session_id, role_id, answers)
    # answers: list of (question_id, answer_record) tuples

    engine.score_question(question_id, answer_record, config)  # per-question
    engine.score_all(answers, config)                         # batch
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger
from utils.validators import clamp_score

from .aggregator import (
    DEFAULT_DIMENSION_WEIGHTS,
    ScoringConfig,
    aggregate_session,
    classify_recommendation,
    compute_question_score,
)
from .consistency import score_consistency
from .dimension_scorers import DimensionResult, score_clarity, score_completeness, score_relevance
from .question_bank_loader import QUESTION_BANK, Question, find_question, questions_for_role
from .scoring_format import (
    DIMENSION_NAMES,
    MODEL_VERSION,
    PIPELINE_VERSION,
    SCHEMA_VERSION,
    DimensionScore,
    ScreeningScoringFormat,
    SessionScore,
)

logger = get_logger("day26_screening_scoring_engine.scoring_engine")


# ---- per-question scoring ---- #


@dataclass
class ScreenAnswerScore:
    """
    Per-question scoring output.

    Aggregated into ScreeningScoringFormat for persistence,
    and fed into aggregate_session() for the session total.
    """

    question_id: str
    raw_answer: str
    dimension_scores: List[DimensionScore]
    overall_score: float  # 0.0-1.0
    weighted_contribution: float  # 0.0-1.0
    scoring_weight: int
    is_mandatory: bool
    expected_answer_type: str
    explanation: str
    warnings: List[str] = field(default_factory=list)
    was_answered: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "raw_answer": self.raw_answer,
            "dimension_scores": [d.to_dict() for d in self.dimension_scores],
            "overall_score": round(self.overall_score, 4),
            "weighted_contribution": round(self.weighted_contribution, 4),
            "scoring_weight": self.scoring_weight,
            "is_mandatory": self.is_mandatory,
            "expected_answer_type": self.expected_answer_type,
            "explanation": self.explanation,
            "warnings": self.warnings,
            "was_answered": self.was_answered,
        }


class ScreeningScoringEngine:
    """
    Scores candidate screening answers across four dimensions and aggregates
    a session-level total.

    Design:
    - Rule-based, LLM-free default. All four dimensions are computed from
      heuristics over the answer text and the extracted slots from
      AnswerUnderstandingEngine (Day 25).
    - Optional LLM re-scoring is a stub (`_call_llm`) — add structured
      JSON-mode prompting to upgrade without changing the public interface.
    - The question bank is loaded once at import time (cached) from
      hr_screening_question_bank/ via question_bank_loader.
    - Outputs are Day 7 envelope-compatible: schema_version, model_version,
      pipeline_version, request_id, generated_at.

    Usage:
        engine = ScreeningScoringEngine()
        result = engine.score_session(
            candidate_id="cand_001",
            job_id="job_backend_01",
            session_id="sess_001",
            role_id="software_engineer",
            answers=[
                (question_id, {
                    "raw_answer": "I have 5 years of experience...",
                    "extracted": {"years_experience": 5.0},
                    "intent_label": "answer",
                    "completeness": 0.9,
                    "warnings": [],
                })
            ],
        )
    """

    def __init__(
        self,
        scoring_config: Optional[ScoringConfig] = None,
        use_llm: bool = False,
    ) -> None:
        """
        Args:
            scoring_config: Tunable dimension weights + thresholds.
                            Defaults to aggregator.DEFAULT_DIMENSION_WEIGHTS.
            use_llm:          If True, attempts LLM re-scoring for clarity/
                               relevance/completeness after the rule-based pass.
                               (Stub: always falls back to rules.)
        """
        self.config = scoring_config or ScoringConfig.with_overrides()
        self.use_llm = use_llm
        logger.info(
            "ScreeningScoringEngine initialized: use_llm=%s weights=%s",
            use_llm,
            self.config.dimension_weights,
        )

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def score_session(
        self,
        candidate_id: str,
        job_id: str,
        session_id: str,
        role_id: str,
        answers: List[Tuple[str, Dict[str, Any]]],
        *,
        request_id: Optional[str] = None,
    ) -> SessionScore:
        """
        Score all answers in one screening session and return a SessionScore.

        Args:
            candidate_id: Unique candidate identifier.
            job_id:       The job this screening is for.
            session_id:   Unique identifier for this call/session.
            role_id:      e.g. "software_engineer" (matches question bank).
            answers:      List of (question_id, answer_record) tuples.
                          Each answer_record should have at minimum:
                            raw_answer: str
                            expected_answer_type: str  (or looked up from bank)
                            category: str              (or looked up from bank)
                            expected_slot: str         (optional)
                            extracted: Dict           (from Day 25)
                            intent_label: str         (from Day 25)
                            warnings: List[str]        (from Day 25)
                            completeness: float        (from Day 25, 0-1)
        """
        rid = request_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Scoring session: candidate=%s session=%s role=%s answers=%d",
            candidate_id,
            session_id,
            role_id,
            len(answers),
        )

        # Run per-question scoring.
        question_records: List[Tuple[str, Dict[str, Any]]] = []
        for qid, rec in answers:
            score_record = self.score_question(qid, rec, self.config)
            question_records.append((qid, score_record.to_dict()))

        # Run consistency check (cross-question).
        session_for_consistency = [
            (
                qid,
                {
                    "category": rec.get("category", ""),
                    "scoring_weight": rec.get("scoring_weight", 1),
                    "raw_answer": rec.get("raw_answer", ""),
                    "extracted": rec.get("extracted", {}),
                },
            )
            for qid, rec in answers
        ]
        consistency_result = score_consistency(session_for_consistency)

        # Aggregate session.
        raw_session = aggregate_session(
            question_records,
            self.config,
            candidate_id=candidate_id,
            job_id=job_id,
            session_id=session_id,
            role_id=role_id,
            request_id=rid,
        )

        # Build per-question ScreeningScoringFormat records for the breakdown.
        breakdowns_list: List[ScreeningScoringFormat] = []
        for qid, rec in question_records:
            breakdowns_list.append(self._build_screening_format(qid, rec, consistency_result))

        # Build SessionScore.
        overall_dim = raw_session.get("overall_dimension_scores", {})
        session = SessionScore(
            candidate_id=candidate_id,
            job_id=job_id,
            session_id=session_id,
            generated_at=now,
            request_id=rid,
            role_id=role_id,
            total_questions=raw_session["total_questions"],
            answered_questions=raw_session["answered_questions"],
            mandatory_unanswered=raw_session["mandatory_unanswered"],
            sum_of_weights_answered=raw_session["sum_of_weights_answered"],
            max_possible_weight=raw_session["max_possible_weight"],
            normalized_score=raw_session["normalized_score"],
            overall_dimension_scores=overall_dim,
            hard_filters_failed=raw_session["hard_filters_failed"],
            recommendation=raw_session["recommendation"],
            consistency=self._consistency_to_format(consistency_result, raw_session),
            breakdown=[self._build_scoring_breakdown(qid, rec) for qid, rec in question_records],
            narrative=raw_session["narrative"],
            warnings=raw_session["warnings"],
        )

        logger.info(
            "Session scored: candidate=%s session=%s score=%.2f recommendation=%s",
            candidate_id,
            session_id,
            session.normalized_score,
            session.recommendation,
        )

        return session

    # ------------------------------------------------------------------ #
    # Per-question scoring
    # ------------------------------------------------------------------ #
    def score_question(
        self,
        question_id: str,
        answer_record: Dict[str, Any],
        config: Optional[ScoringConfig] = None,
    ) -> ScreenAnswerScore:
        """
        Score one question's answer across all four dimensions.

        answer_record must have:
          raw_answer: str
          expected_answer_type: str   (or resolved from question bank)
          category: str                (or resolved from question bank)
          expected_slot: str           (optional, from Day 25 expected_slot)
          extracted: Dict             (from Day 25, slot name -> value)
          intent_label: str           (from Day 25)
          completeness: float         (from Day 25, 0-1)
          warnings: List[str]         (from Day 25)
        """
        cfg = config or self.config

        # Resolve question metadata from the bank.
        q_meta = find_question(question_id)
        if q_meta is None:
            logger.warning(
                "Question %s not found in bank; using answer_record fields.", question_id
            )
            expected_type = str(answer_record.get("expected_answer_type", "text"))
            category = str(answer_record.get("category", "unknown"))
            is_mandatory = bool(answer_record.get("is_mandatory", False))
            scoring_weight = int(answer_record.get("scoring_weight", 1))
            expected_slot = answer_record.get("expected_slot")
        else:
            expected_type = q_meta.expected_answer_type
            category = q_meta.category
            is_mandatory = q_meta.mandatory
            scoring_weight = q_meta.scoring_weight
            expected_slot = None  # not stored in bank

        raw_answer = str(answer_record.get("raw_answer", "") or "")
        extracted: Dict[str, Any] = dict(answer_record.get("extracted") or {})
        intent_label = str(answer_record.get("intent_label") or "answer")
        day25_completeness = float(answer_record.get("completeness", 1.0))
        warnings: List[str] = list(answer_record.get("warnings") or [])

        # ---- Dimension 1: Clarity ----
        clarity_result = score_clarity(raw_answer, expected_type)

        # ---- Dimension 2: Relevance ----
        relevance_result = score_relevance(
            raw_answer,
            expected_type,
            category,
            extracted=extracted,
            intent_label=intent_label,
        )

        # ---- Dimension 3: Completeness ----
        completeness_result = score_completeness(
            raw_answer,
            expected_type,
            expected_slot=expected_slot,
            extracted=extracted,
            intent_label=intent_label,
        )

        # ---- Consistency placeholder ---- (filled in session context)
        consistency_score = 1.0
        consistency_conf = 0.5

        # ---- Dimension scores list ----
        dim_scores_list: List[DimensionScore] = [
            DimensionScore(
                name="clarity",
                score=clamp_score(clarity_result.score),
                confidence=clamp_score(clarity_result.confidence, 0.0, 1.0),
                notes=clarity_result.notes,
            ),
            DimensionScore(
                name="relevance",
                score=clamp_score(relevance_result.score),
                confidence=clamp_score(relevance_result.confidence, 0.0, 1.0),
                notes=relevance_result.notes,
            ),
            DimensionScore(
                name="completeness",
                score=clamp_score(completeness_result.score),
                confidence=clamp_score(completeness_result.confidence, 0.0, 1.0),
                notes=completeness_result.notes,
            ),
            DimensionScore(
                name="consistency",
                score=clamp_score(consistency_score),
                confidence=clamp_score(consistency_conf),
                notes=["Consistency evaluated at session level."],
            ),
        ]

        # ---- Overall question score ----
        dim_score_map = {d.name: d.score for d in dim_scores_list}
        overall_score = compute_question_score(dim_score_map, cfg.dimension_weights)

        # Intent-based overrides: off_topic / no_response => 0.0, objection => <= 0.4
        if intent_label in ("off_topic", "no_response"):
            overall_score = 0.0
        elif intent_label == "objection":
            overall_score = min(overall_score, 0.4)
        elif intent_label == "redirect":
            overall_score = min(overall_score, 0.3)

        # ---- Explanation ----
        explanation = _build_question_explanation(
            question_id=question_id,
            raw_answer=raw_answer,
            overall_score=overall_score,
            dim_scores=dim_scores_list,
            scoring_weight=scoring_weight,
        )

        # ---- was_answered ----
        was_answered = bool(raw_answer.strip()) and intent_label not in ("no_response", "off_topic")

        # ---- warnings ----
        if intent_label in ("objection", "redirect"):
            warnings.append(f"Intent '{intent_label}' noted; review answer quality.")

        return ScreenAnswerScore(
            question_id=question_id,
            raw_answer=raw_answer,
            dimension_scores=dim_scores_list,
            overall_score=round(overall_score, 4),
            weighted_contribution=0.0,  # filled in aggregation
            scoring_weight=scoring_weight,
            is_mandatory=is_mandatory,
            expected_answer_type=expected_type,
            explanation=explanation,
            warnings=warnings,
            was_answered=was_answered,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _build_screening_format(
        self,
        question_id: str,
        record: Dict[str, Any],
        consistency_result: Any,
    ) -> ScreeningScoringFormat:
        """Build a ScreeningScoringFormat from a score_record dict."""
        dim_scores = [
            DimensionScore(
                name=d["name"],
                score=d["score"],
                confidence=d.get("confidence", 0.5),
                notes=d.get("notes", []),
            )
            for d in record.get("dimension_scores", [])
        ]
        return ScreeningScoringFormat(
            question_id=question_id,
            raw_answer=str(record.get("raw_answer", "")),
            dimension_scores=dim_scores,
            overall_score=float(record.get("overall_score", 0.0)),
            weighted_contribution=float(record.get("weighted_contribution", 0.0)),
            scoring_weight=int(record.get("scoring_weight", 1)),
            is_mandatory=bool(record.get("is_mandatory", False)),
            expected_answer_type=str(record.get("expected_answer_type", "text")),
            explanation=str(record.get("explanation", "")),
            warnings=list(record.get("warnings") or []),
            consistency=None,  # set at session level
        )

    def _build_scoring_breakdown(
        self,
        question_id: str,
        record: Dict[str, Any],
    ) -> ScreeningScoringFormat:
        from .scoring_format import ScoringBreakdown

        dim_scores: Dict[str, float] = {}
        for d in record.get("dimension_scores", []):
            dim_scores[d["name"]] = d["score"]

        return ScoringBreakdown(
            question_id=question_id,
            category=str(record.get("category", "unknown")),
            scoring_weight=int(record.get("scoring_weight", 1)),
            is_mandatory=bool(record.get("is_mandatory", False)),
            was_answered=bool(record.get("was_answered", True)),
            overall_score=float(record.get("overall_score", 0.0)),
            weighted_contribution=float(record.get("weighted_contribution", 0.0)),
            dimension_scores=dim_scores,
            explanation=str(record.get("explanation", "")),
        )

    # ------------------------------------------------------------------ #
    # Batch scoring
    # ------------------------------------------------------------------ #
    def score_all(
        self,
        answers: List[Tuple[str, Dict[str, Any]]],
    ) -> List[ScreenAnswerScore]:
        """
        Score a list of (question_id, answer_record) tuples.
        Returns a list of ScreenAnswerScore, one per answer.
        """
        return [self.score_question(qid, rec, self.config) for qid, rec in answers]

    # ------------------------------------------------------------------ #
    # Type conversion helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _consistency_to_format(
        consistency_result: Any,
        raw_session: Dict[str, Any],
    ) -> "ConsistencyResult":
        """
        Convert a DimensionResult (from consistency.py) to a ConsistencyResult
        (from scoring_format.py), so it can be embedded in SessionScore.
        """
        from .scoring_format import ConsistencyResult as CR

        notes = list(getattr(consistency_result, "notes", []) or [])
        flagged: List[str] = []
        # Pull hard-failed question IDs out of the raw session to populate
        # the flagged_questions field.
        for qid in raw_session.get("hard_filters_failed", []):
            flagged.append(qid)
        return CR(
            is_consistent=(getattr(consistency_result, "score", 1.0) >= 0.7 and not flagged),
            score=float(getattr(consistency_result, "score", 1.0)),
            flagged_questions=flagged,
            notes=notes,
        )


# ---- helpers ---- #


def _build_question_explanation(
    question_id: str,
    raw_answer: str,
    overall_score: float,
    dim_scores: List[DimensionScore],
    scoring_weight: int,
) -> str:
    """Build a one-sentence explanation for one question's score."""
    dim_map = {d.name: d.score for d in dim_scores}
    score_pct = f"{overall_score:.0%}"

    labels = {
        "clarity": "clarity",
        "relevance": "relevance",
        "completeness": "completeness",
        "consistency": "consistency",
    }

    sorted_dims = sorted(dim_map.items(), key=lambda x: x[1])
    weakest = sorted_dims[0]
    strongest = sorted_dims[-1]

    parts = [f"Q{question_id} scored {score_pct} (weight {scoring_weight})."]
    parts.append(f"Strongest: {labels.get(weakest[0], strongest[0])} ({strongest[1]:.0%}).")
    parts.append(f"Weakest: {labels.get(weakest[0], weakest[0])} ({weakest[1]:.0%}).")

    if overall_score < 0.5:
        parts.append("Below-average answer; review carefully.")
    elif overall_score >= 0.8:
        parts.append("Strong answer across all dimensions.")

    return " ".join(parts)
