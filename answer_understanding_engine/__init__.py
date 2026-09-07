"""
answer_understanding_engine
-------------------------
Day 25 — Answer Intent & Understanding Engine.

Turns a candidate's free-form reply to a screening question into a
structured semantic object: intent label, slot values, confidence, and
quality flags (off-topic / vague / missing).

Public API:

    from answer_understanding_engine.engine import (
        AnswerUnderstandingEngine,
        StructuredAnswer,
        IntentResult,
        Slot,
    )

    engine = AnswerUnderstandingEngine()
    answer = engine.understand(
        "I have 5 years of experience in Python and Django.",
        question_id="Q-SE-006",
        expected_slot="years_experience",
    )
    print(answer.intent.label)        # "answer"
    print(answer.extracted["years_experience"])  # 5.0
"""

from __future__ import annotations

from .engine import (
    AnswerUnderstandingEngine,
    IntentResult,
    Slot,
    StructuredAnswer,
    _normalize_duration,
    _normalize_money_to_lakhs,
)

__all__ = [
    "AnswerUnderstandingEngine",
    "IntentResult",
    "Slot",
    "StructuredAnswer",
    # helpers (lowercase for internal/advanced use)
    "normalize_duration",
    "normalize_money_to_lakhs",
]

# Public aliases
normalize_duration = _normalize_duration
normalize_money_to_lakhs = _normalize_money_to_lakhs

__version__ = "1.0.0"
