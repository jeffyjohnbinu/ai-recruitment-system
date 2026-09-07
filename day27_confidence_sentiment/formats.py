"""
formats.py
----------
Day 27 deliverable — Zecpath AI Job Portal

Output dataclasses and serialization for confidence & sentiment analysis.
Follows the Day 7 metadata envelope convention: every record carries
schema_version, model_version, pipeline_version, generated_at, request_id.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Module-level versions (also exported from __init__).
SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "day27-v1"
PIPELINE_VERSION = "1.0.0"


# ---- 1. Hesitation pattern record ---- #


@dataclass
class HesitationPattern:
    """A single detected hesitation marker in a candidate's answer."""

    pattern_type: str  # "filler" | "pause" | "repetition" | "repair" | "false_start"
    text: str  # the matched span (e.g., "um", "uh", "I... I think")
    position: int  # character index in the answer (or word index)
    severity: float  # 0.0-1.0, 1.0 = severe
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_type": self.pattern_type,
            "text": self.text,
            "position": self.position,
            "severity": round(self.severity, 3),
            "note": self.note,
        }


# ---- 2. Pace metrics record ---- #


@dataclass
class PaceMetrics:
    """Response length and pace characteristics for one answer."""

    word_count: int
    char_count: int
    sentence_count: int
    avg_sentence_length: float  # words per sentence
    duration_seconds: Optional[float]  # None if not provided
    words_per_second: Optional[float]  # None if no duration
    pace_label: str  # "too_slow" | "slow" | "normal" | "fast" | "too_fast"
    pace_score: float  # 0.0-1.0 (1.0 = ideal pace)
    length_label: str  # "too_short" | "short" | "normal" | "long" | "too_long"
    length_score: float  # 0.0-1.0 (1.0 = ideal length)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word_count": self.word_count,
            "char_count": self.char_count,
            "sentence_count": self.sentence_count,
            "avg_sentence_length": round(self.avg_sentence_length, 2),
            "duration_seconds": (
                round(self.duration_seconds, 2) if self.duration_seconds is not None else None
            ),
            "words_per_second": (
                round(self.words_per_second, 2) if self.words_per_second is not None else None
            ),
            "pace_label": self.pace_label,
            "pace_score": round(self.pace_score, 3),
            "length_label": self.length_label,
            "length_score": round(self.length_score, 3),
            "note": self.note,
        }


# ---- 3. Uncertainty / contradiction signal record ---- #


@dataclass
class UncertaintySignal:
    """A detected hedge, doubt marker, or self-contradiction."""

    signal_type: (
        str  # "hedge" | "doubt" | "self_contradiction" | "mixed_polarity" | "vague_quantifier"
    )
    text: str  # the matched span
    position: int  # character index
    severity: float  # 0.0-1.0
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_type": self.signal_type,
            "text": self.text,
            "position": self.position,
            "severity": round(self.severity, 3),
            "note": self.note,
        }


# ---- 4. Sentiment score record ---- #


@dataclass
class SentimentScore:
    """Positive/negative sentiment and emotion polarity for one answer."""

    polarity: float  # -1.0 (very negative) to +1.0 (very positive)
    positive_score: float  # 0.0-1.0
    negative_score: float  # 0.0-1.0
    neutral_score: float  # 0.0-1.0
    sentiment_label: str  # "positive" | "neutral" | "negative" | "mixed"
    confidence: float  # 0.0-1.0
    emotion_signals: Dict[str, float] = field(default_factory=dict)
    # emotion_signals: e.g., {"enthusiasm": 0.7, "frustration": 0.2, "calm": 0.5}
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "polarity": round(self.polarity, 3),
            "positive_score": round(self.positive_score, 3),
            "negative_score": round(self.negative_score, 3),
            "neutral_score": round(self.neutral_score, 3),
            "sentiment_label": self.sentiment_label,
            "confidence": round(self.confidence, 3),
            "emotion_signals": {k: round(v, 3) for k, v in self.emotion_signals.items()},
            "note": self.note,
        }


# ---- 5. Communication strength indicator ---- #


@dataclass
class CommunicationStrengthIndicator:
    """Composite indicator of how well a candidate communicates."""

    clarity: float  # 0.0-1.0
    confidence: float  # 0.0-1.0
    conviction: float  # 0.0-1.0
    engagement: float  # 0.0-1.0
    professionalism: float  # 0.0-1.0
    overall_strength: float  # 0.0-1.0
    strength_label: str  # "weak" | "developing" | "competent" | "strong" | "exceptional"
    primary_strength: str  # dimension name with the highest score
    primary_weakness: str  # dimension name with the lowest score
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clarity": round(self.clarity, 3),
            "confidence": round(self.confidence, 3),
            "conviction": round(self.conviction, 3),
            "engagement": round(self.engagement, 3),
            "professionalism": round(self.professionalism, 3),
            "overall_strength": round(self.overall_strength, 3),
            "strength_label": self.strength_label,
            "primary_strength": self.primary_strength,
            "primary_weakness": self.primary_weakness,
            "note": self.note,
        }


# ---- 6. Per-answer confidence analysis ---- #


@dataclass
class ConfidenceAnalysis:
    """Confidence and behavioral signals for one answer."""

    question_id: str
    raw_answer: str
    hesitation_patterns: List[HesitationPattern]
    pace_metrics: PaceMetrics
    uncertainty_signals: List[UncertaintySignal]
    sentiment: SentimentScore
    strength_indicator: CommunicationStrengthIndicator

    hesitation_count: int
    uncertainty_count: int
    overall_confidence_score: float  # 0.0-1.0

    generated_at: str
    request_id: str
    schema_version: str = SCHEMA_VERSION
    model_version: str = MODEL_VERSION
    pipeline_version: str = PIPELINE_VERSION
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "pipeline_version": self.pipeline_version,
            "generated_at": self.generated_at,
            "request_id": self.request_id,
            "question_id": self.question_id,
            "raw_answer": self.raw_answer,
            "hesitation_patterns": [h.to_dict() for h in self.hesitation_patterns],
            "pace_metrics": self.pace_metrics.to_dict(),
            "uncertainty_signals": [u.to_dict() for u in self.uncertainty_signals],
            "sentiment": self.sentiment.to_dict(),
            "strength_indicator": self.strength_indicator.to_dict(),
            "hesitation_count": self.hesitation_count,
            "uncertainty_count": self.uncertainty_count,
            "overall_confidence_score": round(self.overall_confidence_score, 3),
            "warnings": self.warnings,
        }


# ---- 7. Session-level behavioral indicators report ---- #


@dataclass
class BehavioralIndicatorsReport:
    """Session-level aggregation of confidence and behavioral signals."""

    candidate_id: str
    job_id: str
    session_id: str
    role_id: str
    total_answers: int
    per_answer: List[ConfidenceAnalysis]

    # Aggregated session-level signals
    session_hesitation_rate: float  # hesitation patterns per answer
    session_uncertainty_rate: float  # uncertainty signals per answer
    session_avg_confidence: float  # 0.0-1.0
    session_avg_strength: float  # 0.0-1.0
    session_sentiment_polarity: float  # -1.0 to +1.0
    session_sentiment_label: str

    # Strength distribution
    strength_distribution: Dict[str, int]  # e.g., {"weak": 0, "competent": 5, "strong": 2}

    # Behavioral red flags
    high_hesitation_answers: List[str]  # question_ids
    high_uncertainty_answers: List[str]
    low_confidence_answers: List[str]
    negative_sentiment_answers: List[str]
    contradictions_detected: List[Dict[str, Any]]

    # Engagement indicators
    avg_response_length: float
    avg_pace_wps: Optional[float]
    avg_filler_ratio: float

    generated_at: str
    request_id: str
    schema_version: str = SCHEMA_VERSION
    model_version: str = MODEL_VERSION
    pipeline_version: str = PIPELINE_VERSION
    narrative: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "pipeline_version": self.pipeline_version,
            "generated_at": self.generated_at,
            "request_id": self.request_id,
            "candidate_id": self.candidate_id,
            "job_id": self.job_id,
            "session_id": self.session_id,
            "role_id": self.role_id,
            "total_answers": self.total_answers,
            "per_answer": [a.to_dict() for a in self.per_answer],
            "session_hesitation_rate": round(self.session_hesitation_rate, 3),
            "session_uncertainty_rate": round(self.session_uncertainty_rate, 3),
            "session_avg_confidence": round(self.session_avg_confidence, 3),
            "session_avg_strength": round(self.session_avg_strength, 3),
            "session_sentiment_polarity": round(self.session_sentiment_polarity, 3),
            "session_sentiment_label": self.session_sentiment_label,
            "strength_distribution": self.strength_distribution,
            "high_hesitation_answers": self.high_hesitation_answers,
            "high_uncertainty_answers": self.high_uncertainty_answers,
            "low_confidence_answers": self.low_confidence_answers,
            "negative_sentiment_answers": self.negative_sentiment_answers,
            "contradictions_detected": self.contradictions_detected,
            "avg_response_length": round(self.avg_response_length, 2),
            "avg_pace_wps": (
                round(self.avg_pace_wps, 2) if self.avg_pace_wps is not None else None
            ),
            "avg_filler_ratio": round(self.avg_filler_ratio, 3),
            "narrative": self.narrative,
            "warnings": self.warnings,
        }


# ---- helpers ---- #


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_request_id() -> str:
    return str(uuid.uuid4())
