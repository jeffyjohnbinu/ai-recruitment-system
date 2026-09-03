"""
speech_to_text
---------------
Speech-to-text integration and transcript normalization for the
AI recruitment pipeline.

Public API:
    STTService: Transcribe audio to text using OpenAI Whisper.
    TranscriptNormalizer: Clean and normalize raw transcripts.
    TranscriptProcessor: End-to-end pipeline (transcribe + normalize).

Exports:
    STTService, TranscriptService  (alias)
    TranscriptNormalizer, Normalizer (alias)
    TranscriptProcessor, Processor (alias)
    TranscriptionResult, NormalizationResult, ProcessedTranscript
"""

from speech_to_text.stt_service import STTService, TranscriptionResult, TranscriptSegment
from speech_to_text.transcript_normalizer import NormalizationResult, TranscriptNormalizer
from speech_to_text.transcript_processor import ProcessedTranscript, TranscriptProcessor

# Convenient aliases
TranscriptService = STTService
Normalizer = TranscriptNormalizer
Processor = TranscriptProcessor

__all__ = [
    # STT
    "STTService",
    "TranscriptService",
    "TranscriptSegment",
    "TranscriptionResult",
    # Normalizer
    "TranscriptNormalizer",
    "Normalizer",
    "NormalizationResult",
    # Processor
    "TranscriptProcessor",
    "Processor",
    "ProcessedTranscript",
]
