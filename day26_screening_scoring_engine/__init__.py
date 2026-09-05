"""
day26_screening_scoring_engine
--------------------------------
Day 26 deliverable — Zecpath AI Job Portal

Objective scoring of candidate screening answers across four orthogonal
dimensions (clarity, relevance, completeness, consistency). Produces a
per-question score breakdown and a single aggregated, explainable final
screening score per candidate session.

Additive-only, consistent with every prior day: consumes
AnswerUnderstandingEngine.StructuredAnswer objects (Day 25) and the
question dataset in hr_screening_question_bank without modifying them.

Public API:
    from day26_screening_scoring_engine import (
        ScreeningScoringEngine,
        ScreenAnswerScore,
        ScoringConfig,
        DEFAULT_SCORING_CONFIG,
        QUESTION_BANK,
        find_question,
        questions_for_role,
        aggregate_session,
        classify_recommendation,
    )
"""

from __future__ import annotations

from .aggregator import (
    DEFAULT_DIMENSION_WEIGHTS,
    ScoringConfig,
    aggregate_session,
    classify_recommendation,
)
from .question_bank_loader import (
    QUESTION_BANK,
    CategoryMeta,
    Question,
    find_question,
    load_question_bank,
    questions_for_role,
)
from .scoring_engine import ScreenAnswerScore, ScreeningScoringEngine
from .scoring_format import (
    ConsistencyResult,
    DimensionScore,
    ScoringBreakdown,
    ScreeningScoringFormat,
    SessionScore,
)


def DEFAULT_SCORING_CONFIG() -> ScoringConfig:
    """Return a fresh default ScoringConfig."""
    return ScoringConfig.with_overrides()


__all__ = [
    # Engine
    "ScreeningScoringEngine",
    "ScreenAnswerScore",
    "ScoringConfig",
    "DEFAULT_SCORING_CONFIG",
    "DEFAULT_DIMENSION_WEIGHTS",
    "aggregate_session",
    "classify_recommendation",
    # Format
    "ScreeningScoringFormat",
    "DimensionScore",
    "ScoringBreakdown",
    "SessionScore",
    "ConsistencyResult",
    # Question bank
    "QUESTION_BANK",
    "Question",
    "CategoryMeta",
    "load_question_bank",
    "find_question",
    "questions_for_role",
]

__version__ = "1.0.0"
__day__ = "Day 26"
