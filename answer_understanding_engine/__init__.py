"""
answer_understanding_engine
-------------------------
Day 25 — Answer Intent & Understanding Engine.

Turns a candidate's free-form reply to a screening question into a
structured semantic object: intent label, slot values, confidence, and
quality flags (off-topic / vague / missing).

Sub-modules:
- engine                  main orchestrator (AnswerUnderstandingEngine)
- intent_classifier       standalone intent classification (6 labels)
- structured_answer_format output dataclasses (StructuredAnswer, Slot, ...)

Public API:

    from answer_understanding_engine import (
        # High-level engine
        AnswerUnderstandingEngine,
        StructuredAnswer,
        # Standalone intent classifier
        classify_intent,
        IntentResult as ICIntentResult,
        INTENT_LABELS,
        # Structured answer format
        Slot,
        AnswerQuality,
    )
"""

from __future__ import annotations

from .engine import AnswerUnderstandingEngine, _normalize_duration, _normalize_money_to_lakhs
from .intent_classifier import INTENT_LABELS, IntentResult, classify_intent
from .structured_answer_format import AnswerQuality, Slot, StructuredAnswer

__all__ = [
    # High-level engine
    "AnswerUnderstandingEngine",
    "StructuredAnswer",
    # Standalone intent classifier
    "classify_intent",
    "IntentResult",
    "INTENT_LABELS",
    # Structured answer format pieces
    "Slot",
    "AnswerQuality",
    # Helpers
    "normalize_duration",
    "normalize_money_to_lakhs",
]

# Public aliases
normalize_duration = _normalize_duration
normalize_money_to_lakhs = _normalize_money_to_lakhs

__version__ = "1.0.0"
