"""
day29/test_conversation_state_machine.py
-------------------------------------------
Tests for the Day 29 conversation state machine.

Run with:
    cd day29 && python -m pytest test_conversation_state_machine.py -v
"""

from __future__ import annotations

import pytest
from conversation_state_machine import CallConfig, ConversationState, ConversationStateMachine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_question(
    question_id: str,
    text: str = "Test question?",
    expected_type: str = "text",
    mandatory: bool = True,
    **kwargs,
) -> dict:
    return {
        "question_id": question_id,
        "category": "test",
        "resolved_text": text,
        "expected_answer_type": expected_type,
        "mandatory": mandatory,
        "scoring_weight": 3,
        "validation": {"rule": "any"},
        "routing": {"on_answer": "next", "on_no_response": "skip"},
        "state": "pending",
        **kwargs,
    }


def _machine(config: CallConfig = CallConfig()) -> ConversationStateMachine:
    m = ConversationStateMachine(config)
    m.start("sess_test", "cand_001", "software_engineer")
    return m


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


def test_start_transitions_to_intro():
    m = _machine()
    assert m.current_state == ConversationState.INTRO


def test_ask_question_transitions_to_awaiting_answer():
    m = _machine()
    m.ask_question(_make_question("Q1"))
    assert m.current_state == ConversationState.AWAITING_ANSWER


def test_answer_transitions_to_answered():
    m = _machine()
    m.ask_question(_make_question("Q1"))
    m.handle_answer("Yes")
    assert m.current_state == ConversationState.ANSWERED
    assert m.answers["Q1"] == "yes"


def test_end_transitions_to_ended():
    m = _machine()
    m.ask_question(_make_question("Q1"))
    m.handle_answer("Yes")
    m.handle_end()
    assert m.current_state == ConversationState.ENDED


def test_escalate_transitions_to_escalated():
    m = _machine()
    m.ask_question(_make_question("Q1"))
    m.handle_escalate("Hard filter failed")
    assert m.current_state == ConversationState.ESCALATED


def test_reschedule_transitions_to_rescheduling():
    m = _machine()
    m.ask_question(_make_question("Q1"))
    m.handle_reschedule()
    assert m.current_state == ConversationState.RESCHEDULING


def test_retry_transitions_to_intro():
    m = _machine()
    m.handle_retry()
    assert m.current_state == ConversationState.INTRO
    assert m.call_attempt_count == 1


# ---------------------------------------------------------------------------
# Silence handling
# ---------------------------------------------------------------------------


def test_silence_reprompts_within_limit():
    m = _machine()
    m.ask_question(_make_question("Q1"))
    # Silence within limit → reprompting
    m.handle_silence()
    assert m.current_state == ConversationState.REPROMPTING
    assert m.reprompt_counts["Q1"] == 1

    # Second silence
    m.handle_reprompt("Please answer.")
    m.handle_silence()
    assert m.current_state == ConversationState.REPROMPTING
    assert m.reprompt_counts["Q1"] == 2

    # Third silence → skip
    m.handle_reprompt("Please answer.")
    m.handle_silence()
    assert m.current_state == ConversationState.SKIPPED
    assert "Q1" in m.skipped_questions


def test_silence_exceeds_max_reprompts():
    config = CallConfig(max_reprompts_per_question=1)
    m = _machine(config)
    m.ask_question(_make_question("Q1"))
    m.handle_silence()  # reprompting
    m.handle_reprompt("Please answer.")
    m.handle_silence()  # skip
    assert m.current_state == ConversationState.SKIPPED


# ---------------------------------------------------------------------------
# Confusion handling
# ---------------------------------------------------------------------------


def test_confusion_reprompts_within_limit():
    m = _machine()
    m.ask_question(_make_question("Q1"))
    m.handle_confusion()
    assert m.current_state == ConversationState.REPROMPTING
    assert m.confusion_counts["Q1"] == 1

    m.handle_reprompt("Let me rephrase.")
    m.handle_confusion()
    assert m.confusion_counts["Q1"] == 2

    m.handle_reprompt("Let me rephrase.")
    m.handle_confusion()
    assert m.current_state == ConversationState.SKIPPED


def test_confusion_exceeds_max():
    config = CallConfig(max_confusion_reprompts=1)
    m = _machine(config)
    m.ask_question(_make_question("Q1"))
    m.handle_confusion()
    m.handle_reprompt("Please clarify.")
    m.handle_confusion()
    assert m.current_state == ConversationState.SKIPPED


# ---------------------------------------------------------------------------
# Repeated answer detection
# ---------------------------------------------------------------------------


def test_repeated_answer_triggers_followup():
    m = _machine()
    m.ask_question(_make_question("Q1"))
    m.handle_answer("I have 3 years of experience")
    assert m.current_state == ConversationState.ANSWERED

    # Same answer again
    m.ask_question(_make_question("Q2"))
    m.handle_answer("I have 3 years of experience")
    assert m.current_state == ConversationState.NEEDS_FOLLOWUP


# ---------------------------------------------------------------------------
# Clarification
# ---------------------------------------------------------------------------


def test_invalid_answer_triggers_clarification():
    config = CallConfig()
    m = _machine(config)
    m.ask_question(_make_question("Q1", expected_type="boolean"))
    m.handle_answer("maybe")  # invalid boolean
    assert m.current_state == ConversationState.NEEDS_CLARIFICATION

    m.handle_clarification("Please answer yes or no.")
    assert m.current_state == ConversationState.AWAITING_ANSWER


# ---------------------------------------------------------------------------
# Follow-up
# ---------------------------------------------------------------------------


def test_followup_question_flow():
    m = _machine()
    m.ask_question(_make_question("Q1"))
    m.handle_answer("Yes")
    assert m.current_state == ConversationState.ANSWERED

    followup = _make_question("Q2", text="Can you elaborate on that?")
    m.handle_followup(followup)
    assert m.current_state == ConversationState.AWAITING_ANSWER


# ---------------------------------------------------------------------------
# Skip & escalate
# ---------------------------------------------------------------------------


def test_skip_marks_question():
    m = _machine()
    m.ask_question(_make_question("Q1"))
    m.handle_skip()
    assert m.current_state == ConversationState.SKIPPED
    assert "Q1" in m.skipped_questions


# ---------------------------------------------------------------------------
# Session summary
# ---------------------------------------------------------------------------


def test_session_summary():
    m = _machine()
    m.ask_question(_make_question("Q1"))
    m.handle_answer("Yes")
    summary = m.get_session_summary()
    assert summary["answers"]["Q1"] == "yes"
    assert summary["current_state"] == "answered"
    assert len(summary["state_history"]) > 0


# ---------------------------------------------------------------------------
# Retry / interrupted
# ---------------------------------------------------------------------------


def test_interrupted_resumes_from_last_question():
    m = _machine()
    m.ask_question(_make_question("Q1"))
    m.handle_answer("Yes")
    m.handle_interrupted()
    assert m.current_state == ConversationState.AWAITING_ANSWER


def test_handle_interrupted_when_idle():
    m = _machine()
    m.handle_interrupted()
    assert m.current_state == ConversationState.INTRO


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_cannot_ask_question_while_ended():
    m = _machine()
    m.ask_question(_make_question("Q1"))
    m.handle_answer("Yes")
    m.handle_end()
    with pytest.raises(ValueError):
        m.ask_question(_make_question("Q2"))


def test_cannot_handle_answer_when_not_awaiting():
    m = _machine()
    with pytest.raises(ValueError):
        m.handle_answer("Yes")


def test_get_current_prompt_for_intro():
    m = _machine()
    assert "screening call" in m.get_current_prompt()


def test_get_current_prompt_for_awaiting_answer():
    m = _machine()
    q = _make_question("Q1", text="What is your name?")
    m.ask_question(q)
    assert m.get_current_prompt() == "What is your name?"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
