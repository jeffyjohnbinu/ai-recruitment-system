"""
scoring_format.py
-----------------
Day 26 deliverable — Zecpath AI Job Portal

Dataclass definitions for the four-dimensional scoring format this
module produces:

  ScreeningScoringFormat     -- per-question, per-dimension scoring record
  ScoringBreakdown           -- per-question aggregation
  SessionScore               -- whole-session aggregation (one candidate / call)
  DimensionScore             -- one dimension's score + explanation
  ConsistencyResult          -- cross-question consistency check result

Metadata standard (applied to every persisted record via `schema_version`,
`model_version`, `pipeline_version`, `candidate_id`, `job_id`, `session_id`,
`generated_at`, `request_id`): follows the Day 7 metadata envelope
convention used by every prior day's storage module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "screening-scoring-engine-1.0.0"
PIPELINE_VERSION = "zecpath-day26"

# Four orthogonal scoring dimensions, each in [0.0, 1.0].
DIMENSION_NAMES: tuple[str, ...] = ("clarity", "relevance", "completeness", "consistency")


@dataclass
class DimensionScore:
    """One dimension's score for one question answer."""

    name: str  # one of DIMENSION_NAMES
    score: float  # 0.0-1.0
    confidence: float  # 0.0-1.0, engine's confidence in the score (not the answer)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConsistencyResult:
    """Cross-question consistency check result."""

    is_consistent: bool
    score: float  # 0.0-1.0
    flagged_questions: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScreeningScoringFormat:
    """Per-question scoring record: all four dimensions for one answer."""

    question_id: str
    raw_answer: str
    dimension_scores: List[DimensionScore]
    overall_score: float  # weighted blend across dimensions, 0.0-1.0
    weighted_contribution: float  # this question's share of session total
    scoring_weight: int  # 1-5, from question bank
    is_mandatory: bool
    expected_answer_type: str
    explanation: str
    warnings: List[str] = field(default_factory=list)
    consistency: Optional[ConsistencyResult] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
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
        }
        if self.consistency is not None:
            d["consistency"] = self.consistency.to_dict()
        return d

    def dimension_score(self, name: str) -> Optional[float]:
        for d in self.dimension_scores:
            if d.name == name:
                return d.score
        return None


@dataclass
class ScoringBreakdown:
    """One question's contribution to the session total."""

    question_id: str
    category: str
    scoring_weight: int
    is_mandatory: bool
    was_answered: bool
    overall_score: float
    weighted_contribution: float
    dimension_scores: Dict[str, float]  # dimension name -> score
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "category": self.category,
            "scoring_weight": self.scoring_weight,
            "is_mandatory": self.is_mandatory,
            "was_answered": self.was_answered,
            "overall_score": round(self.overall_score, 4),
            "weighted_contribution": round(self.weighted_contribution, 4),
            "dimension_scores": {k: round(v, 4) for k, v in self.dimension_scores.items()},
            "explanation": self.explanation,
        }


@dataclass
class SessionScore:
    """Aggregate score across all questions in one screening session."""

    candidate_id: str
    job_id: str
    session_id: str
    generated_at: str
    request_id: str
    role_id: str
    total_questions: int
    answered_questions: int
    mandatory_unanswered: int
    sum_of_weights_answered: float
    max_possible_weight: float
    normalized_score: float  # 0.0-1.0
    overall_dimension_scores: Dict[str, float]  # dimension -> session-avg
    hard_filters_failed: List[str]
    recommendation: str  # "proceed" | "hold" | "reject" | "insufficient_data"
    consistency: ConsistencyResult
    breakdown: List[ScoringBreakdown]
    narrative: str
    warnings: List[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION
    model_version: str = MODEL_VERSION
    pipeline_version: str = PIPELINE_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "pipeline_version": self.pipeline_version,
            "candidate_id": self.candidate_id,
            "job_id": self.job_id,
            "session_id": self.session_id,
            "generated_at": self.generated_at,
            "request_id": self.request_id,
            "role_id": self.role_id,
            "total_questions": self.total_questions,
            "answered_questions": self.answered_questions,
            "mandatory_unanswered": self.mandatory_unanswered,
            "sum_of_weights_answered": round(self.sum_of_weights_answered, 2),
            "max_possible_weight": round(self.max_possible_weight, 2),
            "normalized_score": round(self.normalized_score, 4),
            "overall_dimension_scores": {
                k: round(v, 4) for k, v in self.overall_dimension_scores.items()
            },
            "hard_filters_failed": self.hard_filters_failed,
            "recommendation": self.recommendation,
            "consistency": self.consistency.to_dict(),
            "breakdown": [b.to_dict() for b in self.breakdown],
            "narrative": self.narrative,
            "warnings": self.warnings,
        }
