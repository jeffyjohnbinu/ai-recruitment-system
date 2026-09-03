"""
schema.py
---------
Day 23 deliverable — Zecpath AI Job Portal

Dataclass definitions for the two structured objects this module
produces:

  1. VoiceTranscriptRecord  -- one voice-interview session, fully
     transcribed: every question, every speaker segment, raw +
     normalized text, per-segment ASR confidence and timestamps.

  2. AIScreeningRecord      -- derived per-question screening signals
     computed from a VoiceTranscriptRecord, shaped so it can plug into
     ats_scoring_engine as an additional component the same way Day
     9-12 outputs do (alias-tolerant field names on the consuming side,
     not required here).

Metadata standard (applied to every record via `schema_version`,
`model_version`, `pipeline_version`, `candidate_id`, `job_id`,
`generated_at`, `request_id`): follows the Day 7 metadata envelope
convention used by every prior day's storage module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "voice-screening-engine-1.0.0"
PIPELINE_VERSION = "zecpath-day23"

# Below this ASR confidence, a segment is flagged (never dropped --
# see normalizer.py rule 5: low-confidence segments are kept, flagged).
LOW_CONFIDENCE_THRESHOLD = 0.6


# --------------------------------------------------------------------- #
# Raw transcript layer
# --------------------------------------------------------------------- #
@dataclass
class TranscriptSegment:
    """
    One speaker turn (or ASR-chunked fragment of one) inside a single
    interview question. This is the atomic metadata unit called out in
    the Day 23 brief: candidate_id/job_id live one level up on the
    record; question_id, timestamp, and confidence live here.
    """

    segment_id: str  # e.g. "q1_s1"
    question_id: str
    speaker: str  # "candidate" | "interviewer_ai"
    text_raw: str
    text_normalized: str
    timestamp_start: str  # ISO 8601
    timestamp_end: str  # ISO 8601
    confidence: float  # 0.0-1.0, ASR confidence (not answer quality)
    asr_engine: str = "unknown"
    language: str = "en"
    low_confidence: bool = False
    normalization_notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Derive the flag rather than trust the caller, so it can never
        # drift out of sync with the threshold.
        self.low_confidence = self.confidence < LOW_CONFIDENCE_THRESHOLD

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QuestionTranscript:
    """All segments belonging to one interview question, in order."""

    question_id: str
    question_text: str
    asked_at: str  # ISO 8601
    segments: List[TranscriptSegment] = field(default_factory=list)
    answer_duration_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question_text": self.question_text,
            "asked_at": self.asked_at,
            "segments": [s.to_dict() for s in self.segments],
            "answer_duration_seconds": self.answer_duration_seconds,
        }

    def candidate_text(self) -> str:
        """Concatenated normalized candidate-speech text for this question."""
        return " ".join(
            s.text_normalized for s in self.segments if s.speaker == "candidate"
        ).strip()

    def average_confidence(self) -> float:
        candidate_segments = [s for s in self.segments if s.speaker == "candidate"]
        if not candidate_segments:
            return 0.0
        return round(sum(s.confidence for s in candidate_segments) / len(candidate_segments), 4)


@dataclass
class VoiceTranscriptRecord:
    """
    Top-level per-session transcript object -- the "Voice transcript
    schema" deliverable. One record per (candidate, job, session).
    """

    candidate_id: str
    job_id: str
    session_id: str
    generated_at: str
    request_id: str
    questions: List[QuestionTranscript] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    status: str = "success"  # "success" | "partial" | "failed"
    error: Optional[str] = None
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
            "questions": [q.to_dict() for q in self.questions],
            "warnings": self.warnings,
            "status": self.status,
            "error": self.error,
        }

    def all_segments(self) -> List[TranscriptSegment]:
        return [s for q in self.questions for s in q.segments]

    def low_confidence_segment_count(self) -> int:
        return sum(1 for s in self.all_segments() if s.low_confidence)


# --------------------------------------------------------------------- #
# Derived AI screening layer
# --------------------------------------------------------------------- #
@dataclass
class QuestionScreeningResult:
    """
    Screening signal derived from one QuestionTranscript. Shaped to be
    consumable the same way Day 9-12 component outputs are: plain
    numeric scores + flags, no upstream schema dependency.
    """

    question_id: str
    answer_text: str
    relevance_score: float  # 0.0-1.0
    confidence_avg: float  # 0.0-1.0, mean ASR confidence over the answer
    keywords_matched: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AIScreeningRecord:
    """
    Top-level derived screening object -- the "AI screening data
    structure" deliverable. One record per (candidate, job, session),
    built from a VoiceTranscriptRecord.
    """

    candidate_id: str
    job_id: str
    session_id: str
    generated_at: str
    request_id: str
    per_question: List[QuestionScreeningResult] = field(default_factory=list)
    overall_communication_score: float = 0.0
    warnings: List[str] = field(default_factory=list)
    status: str = "success"
    error: Optional[str] = None
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
            "per_question": [q.to_dict() for q in self.per_question],
            "overall_communication_score": self.overall_communication_score,
            "warnings": self.warnings,
            "status": self.status,
            "error": self.error,
        }
