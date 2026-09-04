"""
answer_understanding_engine/tests/test_intent_classifier.py
------------------------------------------------------------
Tests for the standalone intent_classifier module.

Run with:
    pytest answer_understanding_engine/tests/test_intent_classifier.py -v
"""

from __future__ import annotations

import pytest

from answer_understanding_engine.intent_classifier import INTENT_LABELS, classify_intent

# --------------------------------------------------------------------------- #
# Label coverage                                                              #
# --------------------------------------------------------------------------- #


def test_intent_labels_complete():
    assert set(INTENT_LABELS) == {
        "answer",
        "clarification_request",
        "objection",
        "redirect",
        "off_topic",
        "no_response",
    }


# --------------------------------------------------------------------------- #
# Empty / no_response                                                         #
# --------------------------------------------------------------------------- #


def test_empty_string():
    r = classify_intent("")
    assert r.label == "no_response"
    assert r.confidence >= 0.9


def test_whitespace_only():
    assert classify_intent("   \n\t  ").label == "no_response"


@pytest.mark.parametrize(
    "text",
    [
        "I don't know",
        "idk",
        "not sure",
        "hmm",
        "uh",
        "N/A",
        "none",
        "nothing",
    ],
)
def test_filler(text):
    r = classify_intent(text)
    assert r.label == "no_response", f"Failed for: {text}"


# --------------------------------------------------------------------------- #
# clarification_request                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "What do you mean?",
        "Can you clarify?",
        "Why do you ask?",
        "How long is the interview?",
    ],
)
def test_question_is_clarification(text):
    r = classify_intent(text)
    assert r.label == "clarification_request", f"Failed for: {text}"


# --------------------------------------------------------------------------- #
# redirect                                                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "Can we talk about salary instead?",
        "Let's skip this question.",
        "I'd rather discuss the benefits.",
        "Next question please.",
    ],
)
def test_redirect(text):
    r = classify_intent(text)
    assert r.label == "redirect", f"Failed for: {text}"


# --------------------------------------------------------------------------- #
# objection                                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "I don't want to answer that.",
        "This doesn't feel right.",
        "Why should I tell you?",
        "This is a waste of time.",
    ],
)
def test_objection(text):
    r = classify_intent(text)
    assert r.label == "objection", f"Failed for: {text}"


# --------------------------------------------------------------------------- #
# off_topic                                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "Here is a recipe for biryani.",
        "Check out my YouTube channel!",
        "The cricket score is 245/3.",
        "Bitcoin price is 50k today.",
    ],
)
def test_off_topic(text):
    r = classify_intent(text)
    assert r.label == "off_topic", f"Failed for: {text}"


# --------------------------------------------------------------------------- #
# default: answer                                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "I have 5 years of experience.",
        "Yes, I am willing to relocate.",
        "My expected salary is 18 LPA.",
        "I'm a fresher.",
        "I can join immediately.",
    ],
)
def test_on_topic_default_is_answer(text):
    r = classify_intent(text)
    assert r.label == "answer", f"Failed for: {text}"


# --------------------------------------------------------------------------- #
# IntentResult contract                                                       #
# --------------------------------------------------------------------------- #


def test_intent_result_to_dict():
    r = classify_intent("Hello world")
    d = r.to_dict()
    assert set(d.keys()) == {"label", "confidence", "rationale"}
    assert isinstance(d["confidence"], float)


def test_confidence_bounded():
    for text in ["", "yes", "What?", "no way", "recipe pls", "I don't know"]:
        r = classify_intent(text)
        assert 0.0 <= r.confidence <= 1.0
