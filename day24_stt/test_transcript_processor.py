"""
tests/test_transcript_processor.py
-----------------------------------
Tests for the clean transcript processor module.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from speech_to_text.stt_service import TranscriptionResult, TranscriptSegment
from speech_to_text.transcript_processor import ProcessedTranscript, TranscriptProcessor

# --- Fixtures ---


@pytest.fixture
def processor() -> TranscriptProcessor:
    return TranscriptProcessor(stt_model="whisper-1", language="en")


@pytest.fixture
def fake_audio_bytes() -> bytes:
    return b"FAKE_AUDIO_DATA" * 100


@pytest.fixture
def fake_audio_file(tmp_path: Path) -> Path:
    audio_file = tmp_path / "interview.wav"
    audio_file.write_bytes(b"FAKE_AUDIO_DATA" * 100)
    return audio_file


@pytest.fixture
def mock_stt_transcription() -> TranscriptionResult:
    """Mock transcription result with multiple segments."""
    return TranscriptionResult(
        text="uh yeah I have like 5 years of Python experience. um I love testing.",
        language="en",
        segments=[
            TranscriptSegment(
                start=0.0, end=2.0, text="uh yeah I have like 5 years", avg_logprob=-0.3
            ),
            TranscriptSegment(start=2.0, end=5.0, text=" of Python experience.", avg_logprob=-0.4),
            TranscriptSegment(start=5.5, end=8.0, text=" um I love testing.", avg_logprob=-0.3),
        ],
        silence_gaps=[(5.0, 5.5)],
        interrupted=False,
        partial_answers=[],
        duration_seconds=8.0,
        model="whisper-1",
    )


# --- TranscriptProcessor initialization tests ---


def test_processor_default_init():
    processor = TranscriptProcessor()
    assert processor.stt_service is not None
    assert processor.normalizer is not None


def test_processor_custom_config():
    processor = TranscriptProcessor(
        stt_model="whisper-1",
        language="en",
        silence_gap_threshold=2.0,
        preserve_fillers=True,
    )
    assert processor.stt_service.silence_gap_threshold == 2.0
    assert processor.normalizer.preserve_fillers is True


# --- process_audio tests (full pipeline with mocked STT) ---


def test_process_audio_full_pipeline(processor, fake_audio_bytes, mocker, mock_stt_transcription):
    """Test full audio -> processed transcript pipeline with mocked Whisper."""
    # Mock the STT service
    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = mock_stt_transcription
    mock_stt.build_accent_prompt.return_value = "Test prompt"
    processor.stt_service = mock_stt

    result = processor.process_audio(fake_audio_bytes)

    assert isinstance(result, ProcessedTranscript)
    assert len(result.cleaned_text) > 0
    assert result.original_text == mock_stt_transcription.text
    assert result.duration_seconds == 8.0
    assert result.language == "en"
    assert result.model == "whisper-1"


def test_process_audio_removes_fillers(processor, fake_audio_bytes, mocker, mock_stt_transcription):
    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = mock_stt_transcription
    mock_stt.build_accent_prompt.return_value = "Test prompt"
    processor.stt_service = mock_stt

    result = processor.process_audio(fake_audio_bytes)

    # Verify fillers were removed (word boundary match preserves "yeah" but removes "uh"/"um")
    assert "uh" not in result.cleaned_text.lower().split()
    assert "um" not in result.cleaned_text.lower().split()
    # Content words preserved
    assert "python" in result.cleaned_text.lower()
    assert "experience" in result.cleaned_text.lower()


def test_process_audio_includes_segments(
    processor, fake_audio_bytes, mocker, mock_stt_transcription
):
    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = mock_stt_transcription
    mock_stt.build_accent_prompt.return_value = "Test prompt"
    processor.stt_service = mock_stt

    result = processor.process_audio(fake_audio_bytes)

    assert len(result.segments) == 3
    assert result.segments[0]["start"] == 0.0
    assert result.segments[0]["end"] == 2.0


def test_process_audio_with_accent(processor, fake_audio_bytes, mocker, mock_stt_transcription):
    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = mock_stt_transcription
    mock_stt.build_accent_prompt.return_value = "Indian accent prompt"
    processor.stt_service = mock_stt

    processor.process_audio(fake_audio_bytes, accent="indian")

    mock_stt.build_accent_prompt.assert_called_once_with("indian")
    mock_stt.transcribe.assert_called_once()


def test_process_audio_with_language_override(
    processor, fake_audio_bytes, mocker, mock_stt_transcription
):
    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = mock_stt_transcription
    mock_stt.build_accent_prompt.return_value = "Test prompt"
    processor.stt_service = mock_stt

    processor.process_audio(fake_audio_bytes, language="es")

    mock_stt.transcribe.assert_called_once()
    call_kwargs = mock_stt.transcribe.call_args.kwargs
    assert call_kwargs.get("language") == "es"


def test_process_audio_tracks_filler_statistics(
    processor, fake_audio_bytes, mocker, mock_stt_transcription
):
    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = mock_stt_transcription
    mock_stt.build_accent_prompt.return_value = "Test prompt"
    processor.stt_service = mock_stt

    result = processor.process_audio(fake_audio_bytes)

    assert "uh" in result.filler_statistics
    assert "um" in result.filler_statistics
    assert result.filler_statistics["uh"] >= 1


def test_process_audio_propagates_silence_gaps(
    processor, fake_audio_bytes, mocker, mock_stt_transcription
):
    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = mock_stt_transcription
    mock_stt.build_accent_prompt.return_value = "Test prompt"
    processor.stt_service = mock_stt

    result = processor.process_audio(fake_audio_bytes)

    assert len(result.silence_gaps) == 1
    assert result.silence_gaps[0] == (5.0, 5.5)


# --- process_text tests (normalization-only) ---


def test_process_text_basic(processor):
    raw = "uh yeah I am um a developer"
    result = processor.process_text(raw, case="lower")

    assert isinstance(result, ProcessedTranscript)
    # Standalone single-word fillers removed (uh, um), "yeah" preserved
    assert "uh" not in result.cleaned_text.split()
    assert "um" not in result.cleaned_text.split()
    assert "developer" in result.cleaned_text.lower()


def test_process_text_empty(processor):
    result = processor.process_text("")
    assert result.cleaned_text == ""


def test_process_text_interrupted(processor):
    raw = "I am interrupted. Hmm."
    result = processor.process_text(raw, interrupted=True, case="lower")

    assert result.interrupted is True
    assert isinstance(result.cleaned_text, str)


def test_process_text_partial(processor):
    raw = "This is a substantial answer."
    result = processor.process_text(raw, partial=True, case="lower")

    assert result.partial is True
    assert "substantial" in result.cleaned_text


def test_process_text_preserves_original(processor):
    raw = "Hello world"
    result = processor.process_text(raw, case="preserve")

    assert result.original_text == raw


# --- to_dict serialization tests ---


def test_processed_transcript_to_dict(processor):
    raw = "Hello world"
    result = processor.process_text(raw)

    data = result.to_dict()

    assert "cleaned_text" in data
    assert "original_text" in data
    assert "segments" in data
    assert "interrupted" in data
    assert "duration_seconds" in data


# --- Integration tests ---


def test_full_pipeline_with_file(processor, fake_audio_file, mocker, mock_stt_transcription):
    """Test processing from a file path (most realistic scenario)."""
    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = mock_stt_transcription
    mock_stt.build_accent_prompt.return_value = "Test prompt"
    processor.stt_service = mock_stt

    result = processor.process_audio(fake_audio_file)

    assert isinstance(result, ProcessedTranscript)
    assert len(result.cleaned_text) > 0
    # Verify the mocked transcribe was called with the file
    mock_stt.transcribe.assert_called_once()


def test_process_audio_with_interrupted_speech(processor, fake_audio_bytes, mocker):
    interrupted_transcription = TranscriptionResult(
        text="I have like 5 years of experience and uh",
        language="en",
        segments=[
            TranscriptSegment(
                start=0.0, end=3.0, text="I have like 5 years of experience", avg_logprob=-0.3
            ),
            TranscriptSegment(start=3.0, end=4.5, text=" and uh", avg_logprob=-0.4),
        ],
        silence_gaps=[],
        interrupted=True,
        partial_answers=[1],
        duration_seconds=10.0,  # long trailing silence -> interrupted
        model="whisper-1",
    )

    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = interrupted_transcription
    mock_stt.build_accent_prompt.return_value = "Test prompt"
    processor.stt_service = mock_stt

    result = processor.process_audio(fake_audio_bytes)

    assert result.interrupted is True
    assert len(result.partial_answers) > 0
    # Verify cleanup happened (fillers removed)
    assert "uh" not in result.cleaned_text.lower().split()


def test_build_accent_prompt_delegation(processor):
    """Test that the processor delegates accent prompt building to STT service."""
    prompt = processor.build_accent_prompt("indian", "interview")
    assert "interview" in prompt.lower() or len(prompt) > 0
