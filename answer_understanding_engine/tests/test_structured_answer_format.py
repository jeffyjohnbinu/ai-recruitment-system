"""
answer_understanding_engine/tests/test_structured_answer_format.py
-------------------------------------------------------------------
Tests for the structured_answer_format module.

Run with:
    pytest answer_understanding_engine/tests/test_structured_answer_format.py -v
"""

from __future__ import annotations

import json

from answer_understanding_engine.structured_answer_format import (
    AnswerQuality,
    IntentResult,
    Slot,
    StructuredAnswer,
)

# --------------------------------------------------------------------------- #
# Slot                                                                        #
# --------------------------------------------------------------------------- #


def test_slot_to_dict():
    s = Slot(name="years_experience", value=5.0, raw_span="5 years", confidence=0.9)
    assert s.to_dict() == {
        "name": "years_experience",
        "value": 5.0,
        "raw_span": "5 years",
        "confidence": 0.9,
    }


def test_slot_accepts_list_value():
    s = Slot(name="skills", value=["python", "aws"], raw_span="...", confidence=0.85)
    assert s.value == ["python", "aws"]


# --------------------------------------------------------------------------- #
# IntentResult                                                                #
# --------------------------------------------------------------------------- #


def test_intent_result_to_dict():
    r = IntentResult(label="answer", confidence=0.8, rationale="On-topic.")
    assert r.to_dict() == {
        "label": "answer",
        "confidence": 0.8,
        "rationale": "On-topic.",
    }


# --------------------------------------------------------------------------- #
# AnswerQuality                                                               #
# --------------------------------------------------------------------------- #


def test_quality_defaults():
    q = AnswerQuality()
    assert q.is_off_topic is False
    assert q.is_vague is False
    assert q.is_missing is False
    assert q.completeness == 0.0
    assert q.warnings == []


def test_quality_to_dict_includes_all_fields():
    q = AnswerQuality(is_vague=True, completeness=0.5, warnings=["low confidence"])
    d = q.to_dict()
    assert d["is_vague"] is True
    assert d["completeness"] == 0.5
    assert d["warnings"] == ["low confidence"]


# --------------------------------------------------------------------------- #
# StructuredAnswer — construction                                             #
# --------------------------------------------------------------------------- #


def make_sample():
    return StructuredAnswer(
        question_id="Q-SE-006",
        raw_answer="I have 5 years of experience in Python.",
        intent=IntentResult(label="answer", confidence=0.8, rationale="On-topic."),
        slots=[
            Slot(name="years_experience", value=5.0, raw_span="5 years", confidence=0.9),
            Slot(name="skills", value=["python"], raw_span="Python", confidence=0.85),
        ],
        extracted={"years_experience": 5.0, "skills": ["python"]},
        is_off_topic=False,
        is_vague=False,
        is_missing=False,
        completeness=1.0,
        warnings=[],
    )


def test_structured_answer_fields():
    sa = make_sample()
    assert sa.question_id == "Q-SE-006"
    assert sa.intent.label == "answer"
    assert sa.completeness == 1.0


# --------------------------------------------------------------------------- #
# StructuredAnswer.to_dict round-trip                                         #
# --------------------------------------------------------------------------- #


def test_to_dict_has_required_keys():
    d = make_sample().to_dict()
    expected_keys = {
        "question_id",
        "raw_answer",
        "intent",
        "slots",
        "extracted",
        "is_off_topic",
        "is_vague",
        "is_missing",
        "completeness",
        "warnings",
    }
    assert set(d.keys()) == expected_keys


def test_to_dict_is_json_serializable():
    d = make_sample().to_dict()
    # Must not raise
    serialized = json.dumps(d)
    assert isinstance(serialized, str)
    # Round-trip
    parsed = json.loads(serialized)
    assert parsed["question_id"] == "Q-SE-006"
    assert parsed["extracted"]["years_experience"] == 5.0


def test_to_dict_rounds_completeness():
    sa = make_sample()
    sa.completeness = 0.87654
    d = sa.to_dict()
    assert d["completeness"] == 0.88


# --------------------------------------------------------------------------- #
# StructuredAnswer.from_dict inverse                                          #
# --------------------------------------------------------------------------- #


def test_from_dict_round_trip():
    original = make_sample()
    d = original.to_dict()
    rebuilt = StructuredAnswer.from_dict(d)
    assert rebuilt.question_id == original.question_id
    assert rebuilt.raw_answer == original.raw_answer
    assert rebuilt.intent.label == original.intent.label
    assert rebuilt.intent.confidence == original.intent.confidence
    assert len(rebuilt.slots) == len(original.slots)
    assert rebuilt.extracted == original.extracted
    assert rebuilt.completeness == original.completeness


def test_from_dict_handles_missing_optional_fields():
    d = {
        "question_id": "Q-X",
        "raw_answer": "yes",
        "intent": {"label": "answer", "confidence": 0.8, "rationale": "ok"},
        "slots": [],
        "extracted": {},
    }
    sa = StructuredAnswer.from_dict(d)
    assert sa.is_off_topic is False
    assert sa.completeness == 0.0
    assert sa.warnings == []
