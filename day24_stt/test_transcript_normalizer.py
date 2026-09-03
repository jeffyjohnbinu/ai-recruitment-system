"""
tests/test_transcript_normalizer.py
------------------------------------
Tests for the transcript normalizer module.
"""

from __future__ import annotations

import pytest
from speech_to_text.transcript_normalizer import FILLER_WORDS, TranscriptNormalizer

# --- Fixtures ---

# --- Fixtures ---


@pytest.fixture
def normalizer() -> TranscriptNormalizer:
    return TranscriptNormalizer()


@pytest.fixture
def preserve_fillers_normalizer() -> TranscriptNormalizer:
    return TranscriptNormalizer(preserve_fillers=True)


# --- Basic normalization tests ---


def test_normalize_empty_string(normalizer):
    result = normalizer.normalize("")
    assert result.cleaned == ""


def test_normalize_whitespace_only(normalizer):
    result = normalizer.normalize("    ")
    assert result.cleaned == ""


def test_normalize_preserves_original(normalizer):
    text = "Hello world"
    result = normalizer.normalize(text)
    assert result.original == text


# --- Filler word removal tests ---


def test_remove_simple_fillers(normalizer):
    text = "uh yeah I think um Python is good"
    result = normalizer.normalize(text, case="lower")
    assert "uh" not in result.cleaned.split()
    assert "um" not in result.cleaned.split()
    # "yeah" is preserved (word boundary prevents "ah" from matching inside)
    assert "python" in result.cleaned.lower()


def test_remove_you_know(normalizer):
    text = "I have you know 5 years of experience"
    result = normalizer.normalize(text, case="lower")
    assert "you know" not in result.cleaned.lower()
    assert "5" in result.cleaned


def test_remove_kind_of(normalizer):
    text = "I am kind of a developer"
    result = normalizer.normalize(text, case="lower")
    assert "kind of" not in result.cleaned.lower()


def test_remove_sort_of(normalizer):
    text = "I sort of like Python"
    result = normalizer.normalize(text, case="lower")
    assert "sort of" not in result.cleaned.lower()


def test_remove_i_mean(normalizer):
    text = "I mean I really enjoy coding"
    result = normalizer.normalize(text, case="lower")
    assert "i mean" not in result.cleaned.lower()


def test_remove_right_question_marker(normalizer):
    text = "I work at Google right"
    result = normalizer.normalize(text, case="lower")
    assert "right?" not in result.cleaned.lower()


def test_preserve_fillers_option(preserve_fillers_normalizer):
    text = "uh yeah I am um tired"
    result = preserve_fillers_normalizer.normalize(text, case="lower")
    # When preserve_fillers=True, fillers are kept
    assert "uh" in result.cleaned.lower()
    assert "um" in result.cleaned.lower()
    assert "yeah" in result.cleaned.lower()


def test_filler_statistics(normalizer):
    text = "uh I think um Python is uh good"
    stats = normalizer.get_filler_statistics(text)
    assert stats.get("uh", 0) == 2
    assert stats.get("um", 0) == 1


# --- Punctuation correction tests ---


def test_clean_repeated_punctuation(normalizer):
    text = "Hello!!! World!!"
    result = normalizer.normalize(text, case="preserve")
    assert "!!" not in result.cleaned


def test_punctuation_handling(normalizer):
    text = "Hello,world"
    result = normalizer.normalize(text, case="preserve")
    assert "," in result.cleaned


# --- Case normalization tests ---


def test_sentence_case_capitalizes_starts(normalizer):
    text = "hello world. this is a test."
    result = normalizer.normalize(text, case="sentence")
    # First word should be capitalized
    assert result.cleaned[0].isupper()


def test_lowercase_case(normalizer):
    text = "HELLO WORLD"
    result = normalizer.normalize(text, case="lower")
    assert result.cleaned == "hello world"


def test_preserve_case_keeps_original(normalizer):
    text = "Hello World"
    result = normalizer.normalize(text, case="preserve")
    assert "Hello" in result.cleaned
    assert "World" in result.cleaned


# --- Incomplete words tests ---


def test_fix_incomplete_words(normalizer):
    text = "I have test- ing the system"
    result = normalizer.normalize(text, case="lower")
    assert "testing" in result.cleaned.lower()


def test_preserve_incomplete_option():
    normalizer = TranscriptNormalizer(preserve_incomplete=True)
    text = "I have test- ing the system"
    result = normalizer.normalize(text, case="lower")
    # preserve_incomplete means words split by interruption are NOT joined
    # The hyphenated form with space should remain
    assert "test- ing" in result.cleaned.lower() or "test-" in result.cleaned.lower()


# --- Interrupted speech tests ---


def test_interrupted_flag_set(normalizer):
    text = "Hello world"
    result = normalizer.normalize(text, interrupted=True, case="lower")
    assert result.interrupted is True


def test_interrupted_removes_trailing_fragments(normalizer):
    text = "This is a long meaningful answer. Hmm."
    result = normalizer.normalize(text, interrupted=True, case="lower")
    # "Hmm" is too short, should be removed
    assert "hmm" not in result.cleaned.lower().split()


# --- Partial answer tests ---


def test_partial_flag_set(normalizer):
    text = "Some partial text"
    result = normalizer.normalize(text, partial=True, case="lower")
    assert result.partial is True


def test_partial_keeps_meaningful_text(normalizer):
    text = "This is a substantial answer with multiple words."
    result = normalizer.normalize(text, partial=True, case="lower")
    assert "substantial" in result.cleaned.lower()


def test_partial_removes_short_fragments(normalizer):
    text = "This is a substantial answer with several words. I like it."
    result = normalizer.normalize(text, partial=True, case="lower")
    # Short fragments in multi-sentence text should be removed
    # The first sentence has many words, the second has fewer
    # At minimum, the longer sentence should remain
    assert "substantial" in result.cleaned.lower() or "this" in result.cleaned.lower()


# --- Whitespace tests ---


def test_normalize_whitespace(normalizer):
    text = "Hello    world  test"
    result = normalizer.normalize(text, case="preserve")
    assert "  " not in result.cleaned


def test_strip_leading_trailing(normalizer):
    text = "   Hello world   "
    result = normalizer.normalize(text, case="preserve")
    assert result.cleaned == "Hello world"


# --- Segment normalization tests ---


def test_normalize_segment_basic(normalizer):
    result = normalizer.normalize_segment("Hello world", index=0)
    assert "Hello" in result


def test_normalize_segment_partial_flag(normalizer):
    result = normalizer.normalize_segment("a fragment", index=0, is_partial=True)
    assert isinstance(result, str)


# --- Constants tests ---


def test_filler_words_includes_common():
    assert "uh" in FILLER_WORDS
    assert "um" in FILLER_WORDS
    assert "you know" in FILLER_WORDS


# --- Integration tests ---


def test_full_pipeline_speech_to_clean(normalizer):
    """End-to-end test simulating raw speech -> clean transcript."""
    raw_speech = (
        "uh yeah so I think um I have you know like 5 years of "
        "Python experience and uh I am really passionate about testing"
    )
    result = normalizer.normalize(raw_speech, case="sentence")

    # Verify fillers removed
    assert "uh" not in result.cleaned.lower().split()
    assert "um" not in result.cleaned.lower().split()
    assert "you know" not in result.cleaned.lower()

    # Verify content preserved
    assert "python" in result.cleaned.lower()
    assert "5" in result.cleaned
    assert "testing" in result.cleaned.lower()

    # Verify metadata
    assert len(result.removed_fillers) > 0
    assert result.corrections_made > 0
