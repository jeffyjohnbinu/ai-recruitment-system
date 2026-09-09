"""
day29/conversation_state_machine.py
-----------------------------------
Conversation state machine for the Day 29 AI call flow design.

Implements:
- Silence handling (reprompt ladder)
- Confusion handling (rephrase ladder)
- Repeated answer detection (semantic similarity)
- Fallback question resolution
- Follow-up triggers
- Polite failure and retry logic

Designed to be used by the AI voice call orchestration layer (IVR, LLM agent,
or Twilio/AWS Connect integration).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ConversationState(str, Enum):
    """States for the AI screening call conversation."""

    IDLE = "idle"
    INTRO = "intro"
    AWAITING_ANSWER = "awaiting_answer"
    ANSWERED = "answered"
    REPROMPTING = "reprompting"
    NEEDS_CLARIFICATION = "needs_clarification"
    NEEDS_FOLLOWUP = "needs_followup"
    SKIPPED = "skipped"
    ESCALATED = "escalated"
    RESCHEDULING = "rescheduling"
    ENDED = "ended"
    FAILED = "failed"


@dataclass
class CallConfig:
    """Configuration for silence, retry, and conversation thresholds."""

    silence_timeout_seconds: float = 3.0
    max_reprompts_per_question: int = 2
    max_confusion_reprompts: int = 2
    max_skipped_questions_per_call: int = 3
    retry_after_seconds: int = 3600
    max_call_retry_attempts: int = 3
    semantic_similarity_repeat_threshold: float = 0.85


@dataclass
class CallEvent:
    """Event that triggers a state transition."""

    name: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ConversationStateMachine:
    """
    State machine for AI screening calls.

    Handles the full conversation flow from greeting through ending, including
    edge cases: silence, confusion, repeated answers, and call failures.
    """

    def __init__(self, config: Optional[CallConfig] = None) -> None:
        self.config = config or CallConfig()
        self.current_state = ConversationState.IDLE
        self.session_id: Optional[str] = None
        self.candidate_id: Optional[str] = None
        self.role_id: Optional[str] = None
        self.current_question_id: Optional[str] = None
        self.current_question: Optional[Dict[str, Any]] = None
        self.answers: Dict[str, Any] = {}
        self.skipped_questions: List[str] = []
        self.reprompt_counts: Dict[str, int] = {}
        self.confusion_counts: Dict[str, int] = {}
        self.call_attempt_count = 0
        self.last_event: Optional[CallEvent] = None
        self.state_history: List[Dict[str, Any]] = []
        self._last_answer_fingerprint: Optional[str] = None
        self._previous_answer_fingerprint: Optional[str] = None
        self._last_answer_time: Optional[datetime] = None

    def start(self, session_id: str, candidate_id: str, role_id: str) -> ConversationState:
        """Start a new call session."""
        self.session_id = session_id
        self.candidate_id = candidate_id
        self.role_id = role_id
        self._transition(ConversationState.INTRO, CallEvent("call_started"))
        return self.current_state

    def ask_question(self, question: Dict[str, Any]) -> ConversationState:
        """Set the current question and transition to awaiting_answer."""
        if self.current_state not in (
            ConversationState.INTRO,
            ConversationState.ANSWERED,
            ConversationState.SKIPPED,
            ConversationState.NEEDS_FOLLOWUP,
        ):
            raise ValueError("Cannot ask a question while in state: " + str(self.current_state))

        self.current_question = question
        self.current_question_id = question["question_id"]
        self.reprompt_counts.setdefault(question["question_id"], 0)
        self.confusion_counts.setdefault(question["question_id"], 0)
        self._transition(
            ConversationState.AWAITING_ANSWER,
            CallEvent(
                "question_asked",
                {
                    "question_id": question["question_id"],
                    "question_text": question["resolved_text"],
                },
            ),
        )
        return self.current_state

    def handle_answer(self, raw_answer: str) -> ConversationState:
        """
        Process a candidate answer.

        Returns the next state after validation and routing.
        """
        if self.current_state != ConversationState.AWAITING_ANSWER:
            raise ValueError("Cannot handle answer while not awaiting_answer")

        normalized = raw_answer.strip().lower()
        fingerprint = self._fingerprint(normalized)
        self._last_answer_time = datetime.utcnow()

        # Detect repeated answers (semantic similarity via normalized string match)
        if self._is_repeated_answer(fingerprint):
            self._transition(
                ConversationState.NEEDS_FOLLOWUP, CallEvent("repeated_answer_detected")
            )
            return self.current_state

        # Validate against the question's expected answer type
        validation_result = self._validate_answer(self.current_question, normalized)
        if not validation_result["valid"]:
            self._transition(
                ConversationState.NEEDS_CLARIFICATION,
                CallEvent("invalid_answer", {"reason": validation_result["reason"]}),
            )
            return self.current_state

        # Record the answer and route
        self.answers[self.current_question_id] = normalized
        self._last_answer_fingerprint = fingerprint
        self._transition(
            ConversationState.ANSWERED,
            CallEvent(
                "answer_validated",
                {
                    "question_id": self.current_question_id,
                    "answer": normalized,
                },
            ),
        )
        return self.current_state

    def handle_silence(self) -> ConversationState:
        """Handle silence after the AI has spoken a question."""
        if self.current_state != ConversationState.AWAITING_ANSWER:
            return self.current_state

        reprompt_count = self.reprompt_counts.get(self.current_question_id, 0)
        if reprompt_count < self.config.max_reprompts_per_question:
            self.reprompt_counts[self.current_question_id] = reprompt_count + 1
            self._transition(
                ConversationState.REPROMPTING,
                CallEvent(
                    "silence_detected",
                    {
                        "reprompt_number": reprompt_count + 1,
                    },
                ),
            )
        else:
            self._mark_question_skipped("silence")
        return self.current_state

    def handle_confusion(self) -> ConversationState:
        """Handle confusion (e.g., 'sorry?', 'what do you mean?')."""
        if self.current_state != ConversationState.AWAITING_ANSWER:
            return self.current_state

        confusion_count = self.confusion_counts.get(self.current_question_id, 0)
        if confusion_count < self.config.max_confusion_reprompts:
            self.confusion_counts[self.current_question_id] = confusion_count + 1
            self._transition(
                ConversationState.REPROMPTING,
                CallEvent(
                    "confusion_detected",
                    {
                        "reprompt_number": confusion_count + 1,
                    },
                ),
            )
        else:
            self._mark_question_skipped("confusion")
        return self.current_state

    def handle_reprompt(self, reprompt_text: str) -> ConversationState:
        """Apply a reprompt (silence or confusion) and return to awaiting_answer."""
        if self.current_state != ConversationState.REPROMPTING:
            return self.current_state

        self._transition(
            ConversationState.AWAITING_ANSWER,
            CallEvent(
                "reprompt_applied",
                {
                    "reprompt_text": reprompt_text,
                },
            ),
        )
        return self.current_state

    def handle_clarification(self, clarification_text: str) -> ConversationState:
        """Apply a clarification question for an ambiguous answer."""
        if self.current_state != ConversationState.NEEDS_CLARIFICATION:
            return self.current_state

        self._transition(
            ConversationState.AWAITING_ANSWER,
            CallEvent(
                "clarification_applied",
                {
                    "clarification_text": clarification_text,
                },
            ),
        )
        return self.current_state

    def handle_followup(self, followup_question: Dict[str, Any]) -> ConversationState:
        """Apply a follow-up question triggered by a previous answer."""
        if self.current_state not in (ConversationState.NEEDS_FOLLOWUP, ConversationState.ANSWERED):
            return self.current_state

        self.current_question = followup_question
        self.current_question_id = followup_question["question_id"]
        self._transition(
            ConversationState.AWAITING_ANSWER,
            CallEvent(
                "followup_question_asked",
                {
                    "followup_question_id": followup_question["question_id"],
                    "followup_text": followup_question["resolved_text"],
                },
            ),
        )
        return self.current_state

    def handle_skip(self) -> ConversationState:
        """Skip the current question and continue to the next node."""
        if self.current_state in (
            ConversationState.AWAITING_ANSWER,
            ConversationState.REPROMPTING,
            ConversationState.NEEDS_CLARIFICATION,
        ):
            self._mark_question_skipped("manual")
        return self.current_state

    def handle_escalate(self, reason: str) -> ConversationState:
        """Hand off to a human recruiter."""
        self._transition(ConversationState.ESCALATED, CallEvent("escalated", {"reason": reason}))
        return self.current_state

    def handle_reschedule(self) -> ConversationState:
        """Transition to rescheduling state."""
        self._transition(ConversationState.RESCHEDULING, CallEvent("reschedule_requested"))
        return self.current_state

    def handle_retry(self) -> ConversationState:
        """Retry the call after a failure."""
        self.call_attempt_count += 1
        self._transition(
            ConversationState.INTRO,
            CallEvent(
                "call_retried",
                {
                    "attempt": self.call_attempt_count,
                },
            ),
        )
        return self.current_state

    def handle_interrupted(self) -> ConversationState:
        """Resume from the last answered question after a call drop."""
        if self.current_state == ConversationState.ANSWERED:
            # We already answered; resume by asking the next question
            # (current_question_id holds the last asked question)
            self._transition(
                ConversationState.AWAITING_ANSWER,
                CallEvent(
                    "call_resumed",
                    {
                        "resume_from_question_id": self.current_question_id,
                    },
                ),
            )
        else:
            # Not yet answered; start fresh from intro
            self._transition(
                ConversationState.INTRO,
                CallEvent(
                    "call_resumed",
                    {
                        "resume_from_question_id": self.current_question_id,
                    },
                ),
            )
        return self.current_state

    def handle_end(self, reason: str = "completed") -> ConversationState:
        """End the call gracefully."""
        self._transition(ConversationState.ENDED, CallEvent("call_ended", {"reason": reason}))
        return self.current_state

    def handle_failure(self, error_message: str) -> ConversationState:
        """Handle a call failure (service error, connection drop)."""
        self._transition(
            ConversationState.FAILED,
            CallEvent(
                "call_failed",
                {
                    "error": error_message,
                },
            ),
        )
        return self.current_state

    def get_current_prompt(self) -> str:
        """Return the AI should speak now (based on current state)."""
        if self.current_state == ConversationState.INTRO:
            return "Hi, this is an automated screening call from Zecpath. Is this a good time to talk for about 10 minutes?"
        if self.current_state == ConversationState.AWAITING_ANSWER and self.current_question:
            return self.current_question["resolved_text"]
        if self.current_state == ConversationState.REPROMPTING:
            return "No worries — take your time. Could you please answer the question?"
        if self.current_state == ConversationState.NEEDS_CLARIFICATION:
            return "Could you please clarify your answer?"
        if self.current_state == ConversationState.NEEDS_FOLLOWUP:
            return "Thank you. Could you tell me a bit more about that?"
        return ""

    def get_session_summary(self) -> Dict[str, Any]:
        """Return a summary of the call session."""
        return {
            "session_id": self.session_id,
            "candidate_id": self.candidate_id,
            "role_id": self.role_id,
            "current_state": self.current_state.value,
            "answers": self.answers,
            "skipped_questions": self.skipped_questions,
            "reprompt_counts": self.reprompt_counts,
            "confusion_counts": self.confusion_counts,
            "call_attempt_count": self.call_attempt_count,
            "state_history": self.state_history,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transition(self, new_state: ConversationState, event: CallEvent) -> None:
        self.state_history.append(
            {
                "from": self.current_state.value,
                "to": new_state.value,
                "event": event.name,
                "timestamp": event.timestamp.isoformat(),
            }
        )
        self.current_state = new_state
        self.last_event = event

    def _mark_question_skipped(self, reason: str) -> None:
        if self.current_question_id:
            self.skipped_questions.append(self.current_question_id)
        self._transition(
            ConversationState.SKIPPED,
            CallEvent(
                "question_skipped",
                {
                    "reason": reason,
                    "question_id": self.current_question_id,
                },
            ),
        )

    def _is_repeated_answer(self, fingerprint: str) -> bool:
        """Detect repeated answers via normalized string similarity.

        Compares the current fingerprint against the previous answer only.
        Returns True only if the SAME answer is given twice in a row.
        """
        # No previous answer → not a repeat
        if self._last_answer_fingerprint is None:
            # Store this as the first answer fingerprint
            return False

        # Compare against the PREVIOUS answer fingerprint (not the current one)
        if fingerprint == self._last_answer_fingerprint:
            return True

        # Also detect near-duplicates: one is a substring of the other
        # (e.g., "yes" vs "yes, I do")
        return (
            len(fingerprint) >= 3
            and len(self._last_answer_fingerprint) >= 3
            and (
                fingerprint in self._last_answer_fingerprint
                or self._last_answer_fingerprint in fingerprint
            )
        )

    def _validate_answer(self, question: Dict[str, Any], answer: str) -> Dict[str, Any]:
        """Validate an answer against the question's expected answer type."""
        expected_type = question.get("expected_answer_type", "text")
        value = answer.strip().lower()

        if expected_type == "boolean":
            if value in ("yes", "y", "true", "1", "sure", "definitely"):
                return {"valid": True, "value": True}
            if value in ("no", "n", "false", "0", "no thanks", "not now"):
                return {"valid": True, "value": False}
            return {"valid": False, "reason": "expected_yes_or_no"}

        if expected_type == "number":
            try:
                number = float(value)
                min_val = question.get("validation", {}).get("min", 0)
                max_val = question.get("validation", {}).get("max", 50)
                if min_val <= number <= max_val:
                    return {"valid": True, "value": number}
                return {"valid": False, "reason": "out_of_range"}
            except ValueError:
                return {"valid": False, "reason": "expected_number"}

        if expected_type == "enum":
            allowed = question.get("validation", {}).get("allowed_values", [])
            if value in allowed:
                return {"valid": True, "value": value}
            return {"valid": False, "reason": "invalid_enum"}

        if expected_type == "date":
            # Simplified: accept ISO format or common date formats
            import re

            if re.match(r"^\d{4}-\d{2}-\d{2}$", value) or re.match(
                r"^\d{1,2}/\d{1,2}/\d{2,4}$", value
            ):
                return {"valid": True, "value": value}
            return {"valid": False, "reason": "invalid_date"}

        # text/duration: non-empty is sufficient
        if not value:
            return {"valid": False, "reason": "empty_answer"}
        return {"valid": True, "value": value}

    def _fingerprint(self, text: str) -> str:
        """Normalize text for similarity comparison."""
        return " ".join(text.lower().split())
