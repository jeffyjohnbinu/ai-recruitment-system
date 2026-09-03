"""
Automated test suite for the Voice Screening Transcript Data
Architecture (Day 23).

Run with:
    python -m pytest voice_screening_engine/tests/test_voice_screening.py -v

Or use tests/run_tests.py to also generate a timestamped log file
(deliverable: "Test result logs").
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from voice_screening_engine.normalizer import TranscriptNormalizer  # noqa: E402
from voice_screening_engine.schema import (  # noqa: E402
    LOW_CONFIDENCE_THRESHOLD,
    AIScreeningRecord,
    QuestionScreeningResult,
    QuestionTranscript,
    TranscriptSegment,
    VoiceTranscriptRecord,
)
from voice_screening_engine.storage import TranscriptResultStore  # noqa: E402

OUTPUT_DIR = Path(__file__).parent / "test_outputs"


@pytest.fixture
def normalizer():
    return TranscriptNormalizer(term_lookup={"python": "Python", "aws": "AWS", "sql": "SQL"})


@pytest.fixture
def store():
    return TranscriptResultStore(output_dir=OUTPUT_DIR)


def _make_segment(**overrides) -> TranscriptSegment:
    defaults = dict(
        segment_id="q1_s1",
        question_id="q1",
        speaker="candidate",
        text_raw="raw text",
        text_normalized="Raw text.",
        timestamp_start="2026-09-03T10:00:00+00:00",
        timestamp_end="2026-09-03T10:00:05+00:00",
        confidence=0.9,
        asr_engine="whisper-large-v3",
    )
    defaults.update(overrides)
    return TranscriptSegment(**defaults)


# --------------------------------------------------------------------- #
# Normalizer — rule 1: filler stripping
# --------------------------------------------------------------------- #
def test_normalizer_strips_standalone_filler_tokens(normalizer):
    text, notes = normalizer.normalize("um so I worked on uh a backend service")
    assert "um" not in text.lower().split()
    assert "uh" not in text.lower().split()
    assert any("filler" in n for n in notes)


def test_normalizer_does_not_strip_real_word_like(normalizer):
    text, _ = normalizer.normalize("I would like to improve this system")
    assert "like" in text.lower()


def test_normalizer_strips_filler_phrases(normalizer):
    text, _ = normalizer.normalize("it was, you know, a challenging project")
    assert "you know" not in text.lower()


# --------------------------------------------------------------------- #
# Normalizer — rule 2: pause-based sentence breaks
# --------------------------------------------------------------------- #
def test_normalizer_inserts_break_on_long_pause(normalizer):
    word_timestamps = [
        ("I", 0.0, 0.2),
        ("worked", 0.25, 0.6),
        ("there", 0.65, 1.0),
        ("Then", 2.5, 2.8),  # 1.5s gap -> should force a break before this word
        ("I", 2.85, 3.0),
        ("left", 3.05, 3.3),
    ]
    text, notes = normalizer.normalize(
        "i worked there then i left", word_timestamps=word_timestamps
    )
    assert "there. Then" in text or "there." in text
    assert any("pause-based" in n for n in notes)


def test_normalizer_no_break_without_timestamps(normalizer):
    text, notes = normalizer.normalize("i worked there then i left")
    assert not any("pause-based" in n for n in notes)


# --------------------------------------------------------------------- #
# Normalizer — rule 3: capitalization + term canonicalization
# --------------------------------------------------------------------- #
def test_normalizer_capitalizes_sentence_start(normalizer):
    text, _ = normalizer.normalize("this is my answer")
    assert text.startswith("This")


def test_normalizer_canonicalizes_known_terms(normalizer):
    text, _ = normalizer.normalize("i used python and aws for the sql pipeline")
    assert "Python" in text
    assert "AWS" in text
    assert "SQL" in text


def test_normalizer_without_term_lookup_still_works():
    plain = TranscriptNormalizer()
    text, _ = plain.normalize("i used python and aws")
    # No canonicalization without a lookup, but should not crash and
    # should still capitalize the sentence.
    assert text.startswith("I")


# --------------------------------------------------------------------- #
# Normalizer — rule 4: stutter / repeat collapsing
# --------------------------------------------------------------------- #
def test_normalizer_collapses_immediate_repeats(normalizer):
    text, notes = normalizer.normalize("i i think the the project went well")
    lowered = text.lower()
    assert "i i" not in lowered
    assert "the the" not in lowered
    assert any("repeated" in n for n in notes)


# --------------------------------------------------------------------- #
# Normalizer — edge cases
# --------------------------------------------------------------------- #
def test_normalizer_empty_input_returns_empty():
    normalizer = TranscriptNormalizer()
    text, notes = normalizer.normalize("   ")
    assert text == ""
    assert notes == []


def test_normalizer_adds_terminal_punctuation(normalizer):
    text, _ = normalizer.normalize("this has no ending punctuation")
    assert text.endswith(".")


# --------------------------------------------------------------------- #
# Schema — TranscriptSegment low-confidence flagging (rule 5)
# --------------------------------------------------------------------- #
def test_segment_flags_low_confidence_below_threshold():
    seg = _make_segment(confidence=LOW_CONFIDENCE_THRESHOLD - 0.01)
    assert seg.low_confidence is True


def test_segment_does_not_flag_confidence_at_threshold():
    seg = _make_segment(confidence=LOW_CONFIDENCE_THRESHOLD)
    assert seg.low_confidence is False


def test_segment_never_drops_low_confidence_text():
    # Rule 5: low-confidence segments are flagged, never emptied/dropped.
    seg = _make_segment(
        confidence=0.1, text_raw="mumbled answer", text_normalized="Mumbled answer."
    )
    assert seg.text_raw == "mumbled answer"
    assert seg.text_normalized == "Mumbled answer."
    assert seg.low_confidence is True


# --------------------------------------------------------------------- #
# Schema — QuestionTranscript helpers
# --------------------------------------------------------------------- #
def test_question_candidate_text_excludes_interviewer_segments():
    candidate_seg = _make_segment(
        segment_id="q1_s1", speaker="candidate", text_normalized="My answer."
    )
    interviewer_seg = _make_segment(
        segment_id="q1_s0", speaker="interviewer_ai", text_normalized="Tell me about it."
    )
    q = QuestionTranscript(
        question_id="q1",
        question_text="Tell me about it.",
        asked_at="2026-09-03T10:00:00+00:00",
        segments=[interviewer_seg, candidate_seg],
    )
    assert q.candidate_text() == "My answer."


def test_question_average_confidence_over_candidate_segments_only():
    candidate_a = _make_segment(segment_id="q1_s1", speaker="candidate", confidence=0.8)
    candidate_b = _make_segment(segment_id="q1_s2", speaker="candidate", confidence=0.6)
    interviewer = _make_segment(segment_id="q1_s0", speaker="interviewer_ai", confidence=0.99)
    q = QuestionTranscript(
        question_id="q1",
        question_text="Q",
        asked_at="t",
        segments=[interviewer, candidate_a, candidate_b],
    )
    assert q.average_confidence() == pytest.approx(0.7)


def test_question_average_confidence_no_candidate_segments_is_zero():
    interviewer = _make_segment(speaker="interviewer_ai")
    q = QuestionTranscript(
        question_id="q1", question_text="Q", asked_at="t", segments=[interviewer]
    )
    assert q.average_confidence() == 0.0


# --------------------------------------------------------------------- #
# Schema — VoiceTranscriptRecord helpers + envelope
# --------------------------------------------------------------------- #
def test_transcript_record_carries_metadata_envelope():
    record = VoiceTranscriptRecord(
        candidate_id="c1", job_id="j1", session_id="s1", generated_at="t", request_id="r1"
    )
    assert record.schema_version
    assert record.model_version
    assert record.pipeline_version


def test_transcript_record_low_confidence_segment_count():
    low = _make_segment(segment_id="q1_s1", confidence=0.2)
    high = _make_segment(segment_id="q1_s2", confidence=0.95)
    q = QuestionTranscript(question_id="q1", question_text="Q", asked_at="t", segments=[low, high])
    record = VoiceTranscriptRecord(
        candidate_id="c1",
        job_id="j1",
        session_id="s1",
        generated_at="t",
        request_id="r1",
        questions=[q],
    )
    assert record.low_confidence_segment_count() == 1


def test_transcript_record_to_dict_serializable():
    seg = _make_segment()
    q = QuestionTranscript(question_id="q1", question_text="Q", asked_at="t", segments=[seg])
    record = VoiceTranscriptRecord(
        candidate_id="c1",
        job_id="j1",
        session_id="s1",
        generated_at="t",
        request_id="r1",
        questions=[q],
    )
    d = record.to_dict()
    assert d["candidate_id"] == "c1"
    assert d["questions"][0]["segments"][0]["segment_id"] == "q1_s1"


# --------------------------------------------------------------------- #
# Schema — AIScreeningRecord
# --------------------------------------------------------------------- #
def test_screening_record_to_dict_serializable():
    result = QuestionScreeningResult(
        question_id="q1",
        answer_text="My answer.",
        relevance_score=0.8,
        confidence_avg=0.9,
        keywords_matched=["Python"],
        flags=[],
    )
    record = AIScreeningRecord(
        candidate_id="c1",
        job_id="j1",
        session_id="s1",
        generated_at="t",
        request_id="r1",
        per_question=[result],
        overall_communication_score=0.75,
    )
    d = record.to_dict()
    assert d["per_question"][0]["question_id"] == "q1"
    assert d["overall_communication_score"] == 0.75


# --------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------- #
def test_store_saves_transcript_json(store):
    seg = _make_segment()
    q = QuestionTranscript(question_id="q1", question_text="Q", asked_at="t", segments=[seg])
    record = VoiceTranscriptRecord(
        candidate_id="cand_a",
        job_id="job_a",
        session_id="sess_a",
        generated_at=store.now_iso(),
        request_id=store.new_request_id(),
        questions=[q],
    )
    path = store.save_transcript(record)
    assert Path(path).exists()
    assert Path(path).name == "cand_a__job_a__sess_a.json"


def test_store_saves_screening_json(store):
    result = QuestionScreeningResult(
        question_id="q1", answer_text="A", relevance_score=0.5, confidence_avg=0.5
    )
    record = AIScreeningRecord(
        candidate_id="cand_b",
        job_id="job_b",
        session_id="sess_b",
        generated_at=store.now_iso(),
        request_id=store.new_request_id(),
        per_question=[result],
    )
    path = store.save_screening(record)
    assert Path(path).exists()
    assert Path(path).name == "cand_b__job_b__sess_b.screening.json"


def test_store_save_pair_writes_both_files(store):
    seg = _make_segment()
    q = QuestionTranscript(question_id="q1", question_text="Q", asked_at="t", segments=[seg])
    transcript = VoiceTranscriptRecord(
        candidate_id="cand_c",
        job_id="job_c",
        session_id="sess_c",
        generated_at=store.now_iso(),
        request_id=store.new_request_id(),
        questions=[q],
    )
    result = QuestionScreeningResult(
        question_id="q1", answer_text=q.candidate_text(), relevance_score=0.7, confidence_avg=0.9
    )
    screening = AIScreeningRecord(
        candidate_id="cand_c",
        job_id="job_c",
        session_id="sess_c",
        generated_at=store.now_iso(),
        request_id=store.new_request_id(),
        per_question=[result],
    )
    paths = store.save_pair(transcript, screening)
    assert Path(paths["transcript"]).exists()
    assert Path(paths["screening"]).exists()


def test_store_creates_output_directories(tmp_path):
    nested = tmp_path / "nested" / "outputs"
    TranscriptResultStore(output_dir=nested)
    assert (nested / "transcripts").exists()
    assert (nested / "structured").exists()


def test_store_request_id_unique(store):
    assert store.new_request_id() != store.new_request_id()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
