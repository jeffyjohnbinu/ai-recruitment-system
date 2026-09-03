"""
speech_to_text/stt_service.py
------------------------------
Speech-to-text integration layer built on OpenAI Whisper.

Converts raw voice input (audio bytes/file) into structured, timestamped
transcript segments ready for downstream normalization and AI analysis.

Design:
- Wraps the OpenAI Whisper API behind an isolated `_call_whisper()`
  function so tests can mock the network call.
- Performs voice-activity detection (VAD) using Whisper's segment
  timestamps to identify silence gaps and interrupted speech.
- Supports accent/noise handling via the Whisper `prompt` parameter
  and configurable decoding settings.

Usage:
    service = STTService()
    result = service.transcribe("path/to/audio.wav")
    print(result.text)
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.settings import settings
from utils.logger import get_logger

logger = get_logger("speech_to_text.stt_service")

# Supported audio formats
SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}

# Default silence gap (seconds) considered "interrupted speech"
DEFAULT_SILENCE_GAP_THRESHOLD = 1.5


@dataclass
class TranscriptSegment:
    """A single transcribed speech segment with timing and confidence."""

    start: float  # seconds
    end: float  # seconds
    text: str
    avg_logprob: float = 0.0  # confidence indicator from Whisper


@dataclass
class TranscriptionResult:
    """Full output of a transcription request."""

    text: str  # full concatenated transcript
    language: str | None = None
    segments: list[TranscriptSegment] = field(default_factory=list)
    silence_gaps: list[tuple[float, float]] = field(default_factory=list)
    interrupted: bool = False  # True if speech was likely interrupted
    partial_answers: list[int] = field(default_factory=list)  # segment indices
    duration_seconds: float = 0.0
    model: str = "whisper-1"


def _call_whisper(
    audio_file: io.BytesIO,
    model: str,
    language: str | None,
    prompt: str | None,
) -> dict[str, Any]:
    """
    Isolated network call so tests can monkeypatch this function.

    Returns a dict with keys: text, language, segments, duration.
    """
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)

    kwargs: dict[str, Any] = {
        "model": model,
        "file": audio_file,
        "response_format": "verbose_json",
    }

    if language:
        kwargs["language"] = language

    if prompt:
        kwargs["prompt"] = prompt

    response = client.audio.transcriptions.create(**kwargs)

    # Build segments from Whisper's verbose JSON output
    segments: list[dict[str, Any]] = []
    duration = 0.0

    # Whisper verbose_json returns segments as a list of dicts
    raw_segments = getattr(response, "segments", [])
    for seg in raw_segments:
        segments.append(
            {
                "start": float(seg.get("start", 0)),
                "end": float(seg.get("end", 0)),
                "text": str(seg.get("text", "")),
                "avg_logprob": float(seg.get("avg_logprob", 0.0)),
            }
        )
        duration = max(duration, float(seg.get("end", 0)))

    return {
        "text": str(response.text or ""),
        "language": getattr(response, "language", None),
        "segments": segments,
        "duration_seconds": duration,
    }


def _detect_silence_gaps(
    segments: list[TranscriptSegment],
    threshold: float,
) -> list[tuple[float, float]]:
    """
    Find gaps between consecutive segments larger than `threshold` seconds.

    Returns list of (gap_start, gap_end) tuples representing silence periods.
    """
    if len(segments) < 2:
        return []

    gaps: list[tuple[float, float]] = []

    for i in range(len(segments) - 1):
        current_end = segments[i].end
        next_start = segments[i + 1].start
        gap_duration = next_start - current_end

        if gap_duration >= threshold:
            gaps.append((current_end, next_start))

    return gaps


def _detect_interrupted_speech(
    segments: list[TranscriptSegment],
    total_duration: float,
) -> bool:
    """
    Detect if the transcript was likely interrupted.

    Heuristics:
    - Last segment ends far before total audio duration (unfinished thought).
    - No final punctuation in the last segment.
    """
    if not segments:
        return False

    last_segment = segments[-1]
    trailing_silence = total_duration - last_segment.end

    # If there's more than 3 seconds of silence after the last segment,
    # the speaker may have been cut off
    if trailing_silence > 3.0:
        return True

    # If the last segment has no terminal punctuation, it may be a partial answer
    if last_segment.text and last_segment.text.strip()[-1] not in ".!?":
        # Check if there's meaningful trailing silence (>1s) suggesting cut-off
        if trailing_silence > 0.9:
            return True

    return False


def _detect_partial_answers(
    segments: list[TranscriptSegment],
) -> list[int]:
    """
    Find segments that look like partial/fragmentary answers.

    A segment is considered "partial" if:
    - It doesn't end with terminal punctuation (and is not the last segment)
    - It contains fewer than 4 words (fragment, and is not the last segment)
    """
    partial_indices: list[int] = []

    for i, seg in enumerate(segments):
        text = seg.text.strip()
        if not text:
            continue

        is_last = i == len(segments) - 1

        # No terminal punctuation (but don't flag the very last segment,
        # since speakers often end without strong punctuation)
        if not is_last and text[-1] not in ".!?":
            partial_indices.append(i)

        # Fragment check: very short utterances (not the last segment)
        word_count = len(text.split())
        if not is_last and word_count <= 2:
            partial_indices.append(i)

    return list(dict.fromkeys(partial_indices))  # deduplicate


class STTService:
    """
    Speech-to-text service backed by OpenAI Whisper.

    Supports:
    - Multiple audio formats (wav, mp3, m4a, flac, ogg, webm)
    - Accent/noise handling via prompt parameter
    - Silence gap detection
    - Interrupted speech detection
    - Partial answer detection
    """

    def __init__(
        self,
        model: str = "whisper-1",
        language: str | None = None,
        silence_gap_threshold: float = DEFAULT_SILENCE_GAP_THRESHOLD,
    ) -> None:
        """
        Initialize the STT service.

        Args:
            model: Whisper model to use (default: whisper-1).
            language: Expected language code (e.g., "en", "hi", "es").
                     If None, Whisper auto-detects the language.
            silence_gap_threshold: Gap in seconds between segments
                                   to flag as silence (default: 1.5s).
        """
        self.model = model
        self.language = language
        self.silence_gap_threshold = silence_gap_threshold
        logger.info(
            "STTService initialized: model=%s lang=%s silence_threshold=%.1fs",
            model,
            language,
            silence_gap_threshold,
        )

    def transcribe(
        self,
        audio: bytes | str | Path,
        *,
        prompt: str | None = None,
        language: str | None = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio: Audio input as bytes, file path (str/Path), or Path object.
            prompt: Optional hint to Whisper for accent/dialect correction
                    or noise handling (e.g., "This is a professional interview "
                    "in a corporate setting. Technical terms: Python, API, SQL.").
            language: Override instance language for this call.

        Returns:
            TranscriptionResult with text, segments, silence gaps, and flags.

        Raises:
            FileNotFoundError: If audio is a path that doesn't exist.
            ValueError: If audio format is not supported.
            RuntimeError: If the Whisper API call fails.
        """
        audio_bytes = self._load_audio(audio)
        effective_language = language or self.language

        logger.info(
            "Transcribing audio (%.1f KB), prompt=%s, lang=%s",
            len(audio_bytes) / 1024,
            bool(prompt),
            effective_language,
        )

        # Wrap bytes in a file-like object for the API
        audio_buffer = io.BytesIO(audio_bytes)
        audio_buffer.name = "audio.wav"  # API requires a name attribute

        raw_result = _call_whisper(
            audio_file=audio_buffer,
            model=self.model,
            language=effective_language,
            prompt=prompt,
        )

        # Convert raw dict segments to TranscriptSegment objects
        segments = [
            TranscriptSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
                avg_logprob=seg["avg_logprob"],
            )
            for seg in raw_result["segments"]
        ]

        silence_gaps = _detect_silence_gaps(segments, self.silence_gap_threshold)
        interrupted = _detect_interrupted_speech(segments, raw_result["duration_seconds"])
        partial_indices = _detect_partial_answers(segments)

        result = TranscriptionResult(
            text=raw_result["text"],
            language=raw_result.get("language"),
            segments=segments,
            silence_gaps=silence_gaps,
            interrupted=interrupted,
            partial_answers=partial_indices,
            duration_seconds=raw_result["duration_seconds"],
            model=self.model,
        )

        logger.info(
            "Transcription complete: %.1fs audio -> %d segments, " "interrupted=%s partial=%d",
            result.duration_seconds,
            len(segments),
            interrupted,
            len(partial_indices),
        )

        return result

    def _load_audio(self, audio: bytes | str | Path) -> bytes:
        """Load audio from bytes or a file path, returning raw bytes."""
        if isinstance(audio, bytes):
            return audio

        path = Path(audio)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_AUDIO_FORMATS:
            raise ValueError(
                f"Unsupported audio format: {suffix}. "
                f"Supported: {', '.join(sorted(SUPPORTED_AUDIO_FORMATS))}"
            )

        return path.read_bytes()

    def build_accent_prompt(
        self,
        accent: str,
        context: str = "professional interview",
    ) -> str:
        """
        Build a Whisper prompt for better accent/dialect handling.

        Args:
            accent: Accent or dialect hint (e.g., "Indian English", "British RP").
            context: Additional context for the transcription domain.

        Returns:
            A prompt string to pass to Whisper's `prompt` parameter.
        """
        accent_prompts = {
            "indian": (
                "Common terms: experience, software, engineer, developer, "
                "Python, Java, project, team, company. "
                "Listen carefully for schwa reduction and alveolar taps."
            ),
            "british": (
                "Common terms: programme, optimisation, organise, colour, "
                "realise, practise (noun) vs practice (verb). "
                "Listen for non-rhotic r and raised start vowels."
            ),
            "american": (
                "Common terms: program, optimization, organize, color, "
                "realize, practice. Listen for rhotic r and cot-caught merger."
            ),
            "australian": (
                "Common terms: programme, specialise, realise. "
                "Listen for raised FLEECE vowels and diphthong shifts."
            ),
            "generic": (
                "Listen carefully to all speakers. "
                "Transcribe as spoken, maintaining the original words."
            ),
        }

        base_prompt = accent_prompts.get(accent.lower(), accent_prompts["generic"])

        if context:
            return f"[{context}] {base_prompt}"
        return base_prompt
