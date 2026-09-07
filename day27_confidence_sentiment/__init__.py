"""
day27_confidence_sentiment — Zecpath AI Job Portal

Day 27 deliverable: Confidence & Sentiment Signal Analysis.

Assesses communication quality and behavioral indicators by:
  1. Detecting hesitation patterns (fillers, pauses, repetitions, repairs)
  2. Measuring response length and pace (words-per-second, turn duration)
  3. Identifying positive/negative sentiment (emotion polarity)
  4. Detecting uncertainty and contradictions (hedging, mixed signals)
  5. Creating communication strength indicators (composite confidence score)

All scoring is rule-based by default; LLM stubs are provided for future
enhancement. Follows the same metadata envelope pattern as day26 modules.
"""

from __future__ import annotations

__version__ = "1.0.0"
schema_version = "1.0.0"
model_version = "day27-v1"
pipeline_version = "1.0.0"

from .confidence_analyzer import (
    HesitationPattern,
    PaceMetrics,
    UncertaintySignal,
    analyze_confidence,
    detect_hesitation_patterns,
    detect_uncertainty,
    measure_pace,
)
from .engine import ConfidenceSentimentEngine, SessionConfidenceResult
from .formats import (
    BehavioralIndicatorsReport,
    CommunicationStrengthIndicator,
    ConfidenceAnalysis,
    SentimentScore,
)
from .sentiment_scorer import (
    SentimentResult,
    classify_polarity,
    detect_contradictions,
    score_sentiment,
)

__all__ = [
    "ConfidenceSentimentEngine",
    "SessionConfidenceResult",
    "analyze_confidence",
    "detect_hesitation_patterns",
    "measure_pace",
    "detect_uncertainty",
    "score_sentiment",
    "detect_contradictions",
    "classify_polarity",
    "CommunicationStrengthIndicator",
    "ConfidenceAnalysis",
    "SentimentScore",
    "BehavioralIndicatorsReport",
    "HesitationPattern",
    "PaceMetrics",
    "UncertaintySignal",
    "SentimentResult",
]
