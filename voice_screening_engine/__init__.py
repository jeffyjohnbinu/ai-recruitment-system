"""
Voice Screening Transcript Data Architecture
Day 23 deliverable — Zecpath AI Job Portal

Defines how raw voice-interview conversations are converted into
structured, AI-processable data:
  - VoiceTranscriptRecord: per-session transcript (raw + normalized text,
    per-segment ASR confidence, timestamps)
  - AIScreeningRecord: derived per-question screening signals used by
    downstream scoring (ats_scoring_engine-style components)

Additive-only, consistent with every prior day: this package does not
modify any upstream module. It follows the same Day 7 metadata envelope
convention (schema_version, model_version, pipeline_version,
candidate_id, job_id, generated_at, request_id) used across the
pipeline, and the same two-file JSON storage convention (structured +
human-readable) used since Day 5.
"""

from .normalizer import TranscriptNormalizer
from .schema import (
    AIScreeningRecord,
    QuestionScreeningResult,
    QuestionTranscript,
    TranscriptSegment,
    VoiceTranscriptRecord,
)
from .storage import TranscriptResultStore

__all__ = [
    "TranscriptSegment",
    "QuestionTranscript",
    "VoiceTranscriptRecord",
    "QuestionScreeningResult",
    "AIScreeningRecord",
    "TranscriptNormalizer",
    "TranscriptResultStore",
]
__version__ = "1.0.0"
