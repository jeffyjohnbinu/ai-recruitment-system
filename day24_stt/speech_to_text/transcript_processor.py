"""
speech_to_text/transcript_processor.py
------------------------------------------
Clean transcript processor that wires STT + normalization together.

Provides a single entry point for converting raw audio into clean,
structured text ready for downstream AI analysis (ATS matching,
screening, interview evaluation, etc.).

Usage:
    processor = TranscriptProcessor()
    result = processor.process_audio("path/to/interview.wav")
    print(result.cleaned_text)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from speech_to_text.stt_service import STTService
from speech_to_text.transcript_normalizer import TranscriptNormalizer

from utils.logger import get_logger

logger = get_logger("speech_to_text.transcript_processor")


@dataclass
class ProcessedTranscript:
    """Final output of the full transcript processing pipeline."""

    # Core output
    cleaned_text: str
    original_text: str

    # Segment-level output
    segments: list[dict[str, Any]] = field(default_factory=list)
    filler_statistics: dict[str, int] = field(default_factory=dict)

    # Flags from STT
    interrupted: bool = False
    partial: bool = False  # True if transcript contains partial/fragmentary answers
    partial_answers: list[int] = field(default_factory=list)
    silence_gaps: list[tuple[float, float]] = field(default_factory=list)

    # Normalization metadata
    fillers_removed: list[str] = field(default_factory=list)
    corrections_made: int = 0

    # STT metadata
    duration_seconds: float = 0.0
    language: str | None = None
    model: str = "whisper-1"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the processed transcript to a dictionary."""
        return {
            "cleaned_text": self.cleaned_text,
            "original_text": self.original_text,
            "segments": self.segments,
            "interrupted": self.interrupted,
            "partial_answers": self.partial_answers,
            "silence_gaps": self.silence_gaps,
            "fillers_removed": self.fillers_removed,
            "corrections_made": self.corrections_made,
            "filler_statistics": self.filler_statistics,
            "duration_seconds": self.duration_seconds,
            "language": self.language,
            "model": self.model,
        }


class TranscriptProcessor:
    """
    Clean transcript processor that combines STT transcription
    with text normalization into a single pipeline.

    Usage:
        processor = TranscriptProcessor(
            stt_model="whisper-1",
            language="en",
            silence_gap_threshold=1.5,
        )
        result = processor.process_audio("interview.wav")
        print(result.cleaned_text)
    """

    def __init__(
        self,
        stt_model: str = "whisper-1",
        language: str | None = None,
        silence_gap_threshold: float = 1.5,
        preserve_fillers: bool = False,
        preserve_incomplete: bool = False,
    ) -> None:
        """
        Initialize the transcript processor.

        Args:
            stt_model: Whisper model to use for transcription.
            language: Language code for transcription (e.g., "en").
            silence_gap_threshold: Seconds of silence to flag as gap.
            preserve_fillers: Keep filler words for analysis.
            preserve_incomplete: Keep incomplete/interrupted words.
        """
        self.stt_service = STTService(
            model=stt_model,
            language=language,
            silence_gap_threshold=silence_gap_threshold,
        )
        self.normalizer = TranscriptNormalizer(
            preserve_fillers=preserve_fillers,
            preserve_incomplete=preserve_incomplete,
        )
        self.silence_gap_threshold = silence_gap_threshold

        logger.info(
            "TranscriptProcessor initialized: model=%s lang=%s",
            stt_model,
            language,
        )

    def process_audio(
        self,
        audio: bytes | str | Path,
        *,
        accent: str | None = None,
        language: str | None = None,
    ) -> ProcessedTranscript:
        """
        Full pipeline: transcribe audio -> normalize transcript.

        Args:
            audio: Audio input as bytes, file path, or Path object.
            accent: Accent hint for better Whisper performance
                    (e.g., "indian", "british", "american").
            language: Override language for this call.

        Returns:
            ProcessedTranscript with cleaned text and full metadata.
        """
        logger.info("Processing audio: %s accent=%s", audio, accent)

        # Step 1: Transcribe with STT service
        stt_result = self.stt_service.transcribe(
            audio,
            prompt=self.stt_service.build_accent_prompt(accent) if accent else None,
            language=language,
        )

        # Step 2: Normalize each segment and concatenate
        segment_texts = [seg.text for seg in stt_result.segments]
        normalized_segments: list[str] = []
        all_removed_fillers: list[str] = []
        total_corrections = 0

        for i, seg_text in enumerate(segment_texts):
            is_partial = i in stt_result.partial_answers
            normalized = self.normalizer.normalize(
                seg_text,
                interrupted=stt_result.interrupted,
                partial=is_partial,
                case="sentence",
            )
            normalized_segments.append(normalized.cleaned)
            all_removed_fillers.extend(normalized.removed_fillers)
            total_corrections += normalized.corrections_made

        # Combine normalized segments into final text
        cleaned_text = " ".join(normalized_segments)

        # Step 3: Compute filler statistics from original
        filler_stats = self.normalizer.get_filler_statistics(stt_result.text)

        # Step 4: Build segment-level output
        segments = []
        for seg in stt_result.segments:
            segments.append(
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "avg_logprob": seg.avg_logprob,
                }
            )

        result = ProcessedTranscript(
            cleaned_text=cleaned_text,
            original_text=stt_result.text,
            segments=segments,
            filler_statistics=filler_stats,
            interrupted=stt_result.interrupted,
            partial_answers=stt_result.partial_answers,
            silence_gaps=stt_result.silence_gaps,
            fillers_removed=list(dict.fromkeys(all_removed_fillers)),
            corrections_made=total_corrections,
            duration_seconds=stt_result.duration_seconds,
            language=stt_result.language,
            model=stt_result.model,
        )

        logger.info(
            "Transcript processing complete: %.1fs audio, " "%d segments, %d corrections",
            result.duration_seconds,
            len(segments),
            total_corrections,
        )

        return result

    def process_text(
        self,
        raw_text: str,
        *,
        interrupted: bool = False,
        partial: bool = False,
        case: str = "sentence",
    ) -> ProcessedTranscript:
        """
        Normalize already-transcribed text (skip STT step).

        Useful when transcription was done externally or via a different
        service, and only cleaning/normalization is needed.

        Args:
            raw_text: Raw transcript text.
            interrupted: Whether the transcript was interrupted.
            partial: Whether the transcript has partial answers.
            case: Case normalization mode.

        Returns:
            ProcessedTranscript with cleaned text.
        """
        logger.info("Processing raw text: %d chars", len(raw_text))

        normalized = self.normalizer.normalize(
            raw_text,
            interrupted=interrupted,
            partial=partial,
            case=case,
        )

        result = ProcessedTranscript(
            cleaned_text=normalized.cleaned,
            original_text=raw_text,
            fillers_removed=normalized.removed_fillers,
            corrections_made=normalized.corrections_made,
            interrupted=interrupted,
            partial=partial,
        )

        return result

    def build_accent_prompt(self, accent: str, context: str = "") -> str:
        """Delegates to STT service to build accent prompts."""
        return self.stt_service.build_accent_prompt(accent, context)
