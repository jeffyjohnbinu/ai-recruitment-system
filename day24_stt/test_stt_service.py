"""
tests/test_stt_service.py
--------------------------
Tests for the STT service module.
Network calls (OpenAI Whisper) are mocked via `pytest-mock`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from speech_to_text.stt_service import (
    SUPPORTED_AUDIO_FORMATS,
    STTService,
    TranscriptionResult,
    TranscriptSegment,
    _detect_interrupted_speech,
    _detect_partial_answers,
    _detect_silence_gaps,
)

# --- Fixtures ---


@pytest.fixture
def stt_service() -> STTService:
    return STTService(model="whisper-1", language="en")


@pytest.fixture
def fake_audio_bytes() -> bytes:
    """Pretend audio bytes (won't be actually decoded)."""
    return b"FAKE_AUDIO_DATA" * 100


@pytest.fixture
def fake_audio_file(tmp_path: Path) -> Path:
    """Create a fake audio file on disk."""
    audio_file = tmp_path / "test_audio.wav"
    audio_file.write_bytes(b"FAKE_AUDIO_DATA" * 100)
    return audio_file


@pytest.fixture
def mock_whisper_response() -> dict:
    """Mocked Whisper verbose_json response."""
    return {
        "text": "Hello, this is a test transcription.",
        "language": "en",
        "segments": [
            {
                "start": 0.0,
                "end": 1.5,
                "text": "Hello, this is a test",
                "avg_logprob": -0.3,
            },
            {
                "start": 1.5,
                "end": 3.0,
                "text": " transcription.",
                "avg_logprob": -0.4,
            },
        ],
        "duration_seconds": 3.0,
    }


# --- Whisper API mock helper ---


def _patch_whisper(mocker, return_value: dict) -> None:
    """Patch the `_call_whisper` function inside stt_service."""
    mocker.patch(
        "speech_to_text.stt_service._call_whisper",
        return_value=return_value,
    )


# --- STTService.transcribe tests ---


def test_transcribe_from_bytes(stt_service, fake_audio_bytes, mocker, mock_whisper_response):
    _patch_whisper(mocker, mock_whisper_response)
    result = stt_service.transcribe(fake_audio_bytes)

    assert isinstance(result, TranscriptionResult)
    assert result.text == "Hello, this is a test transcription."
    assert result.language == "en"
    assert len(result.segments) == 2
    assert result.duration_seconds == 3.0


def test_transcribe_from_file_path(stt_service, fake_audio_file, mocker, mock_whisper_response):
    _patch_whisper(mocker, mock_whisper_response)
    result = stt_service.transcribe(fake_audio_file)

    assert isinstance(result, TranscriptionResult)
    assert result.text == "Hello, this is a test transcription."


def test_transcribe_from_string_path(stt_service, fake_audio_file, mocker, mock_whisper_response):
    _patch_whisper(mocker, mock_whisper_response)
    result = stt_service.transcribe(str(fake_audio_file))

    assert isinstance(result, TranscriptionResult)
    assert len(result.segments) == 2


def test_transcribe_unsupported_format_raises(stt_service, tmp_path):
    bad_file = tmp_path / "test.txt"
    bad_file.write_bytes(b"not audio")
    with pytest.raises(ValueError, match="Unsupported audio format"):
        stt_service.transcribe(bad_file)


def test_transcribe_nonexistent_file_raises(stt_service, tmp_path):
    missing = tmp_path / "missing.wav"
    with pytest.raises(FileNotFoundError):
        stt_service.transcribe(missing)


def test_transcribe_preserves_accents_via_prompt(
    stt_service, fake_audio_bytes, mocker, mock_whisper_response
):
    _patch_whisper(mocker, mock_whisper_response)

    # Test with accent prompt
    accent_prompt = stt_service.build_accent_prompt("indian", "interview")
    result = stt_service.transcribe(fake_audio_bytes, prompt=accent_prompt)
    assert result is not None


# --- Silence gap detection tests ---


def test_detect_silence_gaps_no_gaps():
    segments = [
        TranscriptSegment(start=0.0, end=1.0, text="A"),
        TranscriptSegment(start=1.2, end=2.0, text="B"),
    ]
    gaps = _detect_silence_gaps(segments, threshold=1.5)
    assert len(gaps) == 0


def test_detect_silence_gaps_finds_gaps():
    segments = [
        TranscriptSegment(start=0.0, end=1.0, text="A"),
        TranscriptSegment(start=3.0, end=4.0, text="B"),
    ]
    gaps = _detect_silence_gaps(segments, threshold=1.5)
    assert len(gaps) == 1
    assert gaps[0] == (1.0, 3.0)


def test_detect_silence_gaps_multiple_gaps():
    segments = [
        TranscriptSegment(start=0.0, end=1.0, text="A"),
        TranscriptSegment(start=3.0, end=4.0, text="B"),
        TranscriptSegment(start=4.5, end=5.5, text="C"),
        TranscriptSegment(start=8.0, end=9.0, text="D"),
    ]
    gaps = _detect_silence_gaps(segments, threshold=1.5)
    assert len(gaps) == 2


def test_detect_silence_gaps_single_segment():
    segments = [TranscriptSegment(start=0.0, end=1.0, text="A")]
    gaps = _detect_silence_gaps(segments, threshold=1.5)
    assert len(gaps) == 0


# --- Interrupted speech detection tests ---


def test_detect_interrupted_speech_long_trailing_silence():
    segments = [TranscriptSegment(start=0.0, end=1.0, text="A.")]
    assert _detect_interrupted_speech(segments, total_duration=5.0) is True


def test_detect_interrupted_speech_complete():
    segments = [TranscriptSegment(start=0.0, end=2.0, text="Complete sentence.")]
    assert _detect_interrupted_speech(segments, total_duration=2.0) is False


def test_detect_interrupted_speech_no_punctuation_short_silence():
    segments = [TranscriptSegment(start=0.0, end=2.0, text="no period")]
    # 1s trailing silence, no terminal punctuation -> interrupted
    assert _detect_interrupted_speech(segments, total_duration=3.0) is True


def test_detect_interrupted_speech_empty():
    assert _detect_interrupted_speech([], total_duration=0.0) is False


# --- Partial answer detection tests ---


def test_detect_partial_answers_finds_fragments():
    segments = [
        TranscriptSegment(start=0.0, end=0.5, text="Yeah", avg_logprob=-0.5),
        TranscriptSegment(start=0.6, end=3.0, text="That is a full answer.", avg_logprob=-0.3),
    ]
    partial = _detect_partial_answers(segments)
    assert 0 in partial  # First segment is a fragment


def test_detect_partial_answers_no_punctuation():
    segments = [
        TranscriptSegment(
            start=0.0, end=1.0, text="A complete thought no period", avg_logprob=-0.3
        ),
        TranscriptSegment(start=1.0, end=2.0, text="Another", avg_logprob=-0.3),
    ]
    partial = _detect_partial_answers(segments)
    assert 0 in partial


def test_detect_partial_answers_complete():
    segments = [
        TranscriptSegment(start=0.0, end=1.0, text="First full answer.", avg_logprob=-0.3),
        TranscriptSegment(start=1.0, end=2.0, text="Second full answer.", avg_logprob=-0.3),
    ]
    partial = _detect_partial_answers(segments)
    assert len(partial) == 0


# --- Accent prompt tests ---


def test_build_accent_prompt_indian():
    service = STTService()
    prompt = service.build_accent_prompt("indian", "interview")
    assert "interview" in prompt.lower() or "interview" in prompt
    assert "python" in prompt.lower() or "engineer" in prompt.lower()


def test_build_accent_prompt_british():
    service = STTService()
    prompt = service.build_accent_prompt("british")
    assert "programme" in prompt.lower() or "optimisation" in prompt.lower()


def test_build_accent_prompt_generic_fallback():
    service = STTService()
    prompt = service.build_accent_prompt("unknown_accent")
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_build_accent_prompt_with_context():
    service = STTService()
    prompt = service.build_accent_prompt("indian", "technical interview")
    assert "technical interview" in prompt.lower() or "interview" in prompt.lower()


# --- Configuration tests ---


def test_stt_service_default_config():
    service = STTService()
    assert service.model == "whisper-1"
    assert service.language is None
    assert service.silence_gap_threshold == 1.5


def test_stt_service_custom_config():
    service = STTService(
        model="whisper-1",
        language="en",
        silence_gap_threshold=2.0,
    )
    assert service.model == "whisper-1"
    assert service.language == "en"
    assert service.silence_gap_threshold == 2.0


def test_supported_audio_formats_contains_common():
    assert ".wav" in SUPPORTED_AUDIO_FORMATS
    assert ".mp3" in SUPPORTED_AUDIO_FORMATS
    assert ".m4a" in SUPPORTED_AUDIO_FORMATS
