"""
conversation_simulation.py
--------------------------
Simulates AI screening calls to validate screening system performance.

Integrates the Day 29 ConversationStateMachine with the Day 26
ScreeningScoringEngine to simulate full screening conversations
and capture scoring results for validation.
"""

from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import Day 26 components
from day26_screening_scoring_engine.scoring_engine import ScreeningScoringEngine
from day26_screening_scoring_engine.scoring_format import SessionScore

# Import Day 29 components
from day29.conversation_state_machine import CallConfig, ConversationState, ConversationStateMachine
from utils.logger import get_logger

logger = get_logger("day30_screening_testing.conversation_simulation")


@dataclass
class SimulatedCallResult:
    """Result of a single simulated screening call."""

    call_id: str
    candidate_id: str
    job_id: str
    session_id: str
    role_id: str
    initial_state: str
    final_state: str
    questions_asked: int
    questions_answered: int
    questions_skipped: int
    reprompts_given: int
    clarifications_given: int
    followups_given: int
    answers: Dict[str, Any] = field(default_factory=dict)
    state_history: List[Dict[str, Any]] = field(default_factory=list)
    scoring_result: Optional[SessionScore] = None
    intent_distribution: Dict[str, int] = field(default_factory=dict)
    call_duration_seconds: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class SimulationBatchResult:
    """Results of a batch of simulated screening calls."""

    batch_id: str
    total_calls: int
    successful_calls: int
    failed_calls: int
    average_questions_asked: float
    average_questions_answered: float
    average_reprompts: float
    average_call_duration: float
    results: List[SimulatedCallResult] = field(default_factory=list)


class ConversationSimulator:
    """
    Simulates AI screening calls for testing the screening system.

    Integrates:
    - Day 29 ConversationStateMachine for conversation flow
    - Day 26 ScreeningScoringEngine for answer scoring
    - Day 25 intent detection for response classification

    Usage:
        simulator = ConversationSimulator()
        result = simulator.simulate_call(
            candidate_id="cand_001",
            job_id="job_001",
            role_id="software_engineer",
            candidate_answers=["Yes", "5 years", "MERN stack"],
        )
    """

    def __init__(
        self,
        scoring_engine: Optional[ScreeningScoringEngine] = None,
        call_config: Optional[CallConfig] = None,
    ) -> None:
        """
        Args:
            scoring_engine: ScreeningScoringEngine instance.
                           Created if not provided.
            call_config: Configuration for silence, retry, and conversation thresholds.
        """
        self.scoring_engine = scoring_engine or ScreeningScoringEngine()
        self.call_config = call_config or CallConfig()
        self._call_results: List[SimulatedCallResult] = []

    def simulate_call(
        self,
        candidate_id: str,
        job_id: str,
        role_id: str,
        candidate_answers: List[str],
        questions: Optional[List[Dict[str, Any]]] = None,
        session_id: Optional[str] = None,
        call_id: Optional[str] = None,
    ) -> SimulatedCallResult:
        """
        Simulate a single AI screening call.

        Args:
            candidate_id: Unique candidate identifier.
            job_id: Job identifier for this screening.
            role_id: Role identifier (e.g., "software_engineer").
            candidate_answers: Predefined answers from the candidate.
            questions: Optional list of questions to ask.
                      Uses default question bank if not provided.
            session_id: Optional session identifier.
            call_id: Optional call identifier.

        Returns:
            SimulatedCallResult with full conversation and scoring data.
        """
        call_id = call_id or f"call_{uuid.uuid4().hex[:8]}"
        session_id = session_id or f"sess_{uuid.uuid4().hex[:8]}"

        logger.info(
            "Starting simulated call: call=%s candidate=%s role=%s",
            call_id,
            candidate_id,
            role_id,
        )

        # Initialize conversation state machine
        machine = ConversationStateMachine(config=self.call_config)
        machine.start(session_id=session_id, candidate_id=candidate_id, role_id=role_id)

        # Get questions for this role
        if questions is None:
            questions = self._get_default_questions(role_id)

        # Track conversation metrics
        questions_asked = 0
        questions_answered = 0
        questions_skipped = 0
        reprompts_given = 0
        clarifications_given = 0
        followups_given = 0
        answers: Dict[str, Any] = {}
        intent_distribution: Dict[str, int] = {}
        start_time = datetime.now(timezone.utc)

        try:
            # Simulate conversation flow
            answer_index = 0
            for question in questions:
                if machine.current_state == ConversationState.ENDED:
                    break

                # Ask question
                machine.ask_question(question)
                questions_asked += 1

                # Get candidate answer (simulated)
                if answer_index < len(candidate_answers):
                    raw_answer = candidate_answers[answer_index]
                    answer_index += 1

                    # Handle answer
                    prev_state = machine.current_state
                    machine.handle_answer(raw_answer)

                    # Track state transitions
                    if machine.current_state == ConversationState.ANSWERED:
                        questions_answered += 1
                        answers[question["question_id"]] = raw_answer

                        # Track intent distribution
                        intent = self._detect_intent(raw_answer)
                        intent_distribution[intent] = intent_distribution.get(intent, 0) + 1

                    elif machine.current_state == ConversationState.NEEDS_CLARIFICATION:
                        clarifications_given += 1
                        machine.handle_clarification("I need to clarify that.")
                        if answer_index < len(candidate_answers):
                            raw_clar = candidate_answers[answer_index]
                            answer_index += 1
                            machine.handle_answer(raw_clar)
                            if machine.current_state == ConversationState.ANSWERED:
                                questions_answered += 1
                                answers[question["question_id"]] = raw_clar
                            else:
                                machine.handle_skip()
                                questions_skipped += 1
                        else:
                            machine.handle_skip()
                            questions_skipped += 1

                    elif machine.current_state == ConversationState.SKIPPED:
                        questions_skipped += 1

                    elif machine.current_state == ConversationState.NEEDS_FOLLOWUP:
                        followups_given += 1
                        followup = self._create_followup(question)
                        machine.handle_followup(followup)
                        if answer_index < len(candidate_answers):
                            raw_fol = candidate_answers[answer_index]
                            answer_index += 1
                            machine.handle_answer(raw_fol)
                            if machine.current_state == ConversationState.ANSWERED:
                                questions_answered += 1
                                answers[question["question_id"]] = raw_fol
                            else:
                                machine.handle_skip()
                                questions_skipped += 1
                        else:
                            machine.handle_skip()
                            questions_skipped += 1

                    elif machine.current_state == ConversationState.REPROMPTING:
                        reprompts_given += 1
                        machine.handle_reprompt("Could you please answer?")
                        if answer_index < len(candidate_answers):
                            raw_rep = candidate_answers[answer_index]
                            answer_index += 1
                            machine.handle_answer(raw_rep)
                            if machine.current_state == ConversationState.ANSWERED:
                                questions_answered += 1
                                answers[question["question_id"]] = raw_rep
                            else:
                                machine.handle_skip()
                                questions_skipped += 1
                        else:
                            machine.handle_skip()
                            questions_skipped += 1

                    if machine.current_state not in (
                        ConversationState.ANSWERED,
                        ConversationState.SKIPPED,
                    ):
                        machine.handle_skip()
                        questions_skipped += 1

                else:
                    # No more answers provided
                    machine.handle_skip()
                    questions_skipped += 1

            # End the call
            machine.handle_end("completed")

            # Calculate call duration
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            # Score the session
            scoring_result = self._score_session(
                candidate_id=candidate_id,
                job_id=job_id,
                session_id=session_id,
                role_id=role_id,
                answers=answers,
                questions=questions,
            )

            # Create result
            result = SimulatedCallResult(
                call_id=call_id,
                candidate_id=candidate_id,
                job_id=job_id,
                session_id=session_id,
                role_id=role_id,
                initial_state=ConversationState.IDLE.value,
                final_state=machine.current_state.value,
                questions_asked=questions_asked,
                questions_answered=questions_answered,
                questions_skipped=questions_skipped,
                reprompts_given=reprompts_given,
                clarifications_given=clarifications_given,
                followups_given=followups_given,
                answers=answers,
                state_history=machine.state_history.copy(),
                scoring_result=scoring_result,
                intent_distribution=intent_distribution,
                call_duration_seconds=duration,
                success=True,
            )

            self._call_results.append(result)
            logger.info(
                "Simulated call completed: call=%s score=%.2f recommendation=%s",
                call_id,
                scoring_result.normalized_score if scoring_result else 0,
                scoring_result.recommendation if scoring_result else "N/A",
            )

            return result

        except Exception as e:
            logger.error("Simulated call failed: call=%s error=%s", call_id, str(e))
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            result = SimulatedCallResult(
                call_id=call_id,
                candidate_id=candidate_id,
                job_id=job_id,
                session_id=session_id,
                role_id=role_id,
                initial_state=ConversationState.IDLE.value,
                final_state=machine.current_state.value if machine else "failed",
                questions_asked=questions_asked,
                questions_answered=questions_answered,
                questions_skipped=questions_skipped,
                reprompts_given=reprompts_given,
                clarifications_given=clarifications_given,
                followups_given=followups_given,
                answers=answers,
                state_history=machine.state_history.copy() if machine else [],
                scoring_result=None,
                intent_distribution=intent_distribution,
                call_duration_seconds=duration,
                success=False,
                error_message=str(e),
            )

            self._call_results.append(result)
            return result

    def simulate_batch(
        self,
        test_cases: List[Dict[str, Any]],
    ) -> SimulationBatchResult:
        """
        Simulate a batch of screening calls.

        Args:
            test_cases: List of test case dictionaries with:
                       - candidate_id
                       - job_id
                       - role_id
                       - candidate_answers
                       - expected_outcome (optional)

        Returns:
            SimulationBatchResult with aggregated results.
        """
        batch_id = f"batch_{uuid.uuid4().hex[:8]}"
        results: List[SimulatedCallResult] = []

        logger.info("Starting batch simulation: batch=%s count=%d", batch_id, len(test_cases))

        for test_case in test_cases:
            result = self.simulate_call(
                candidate_id=test_case["candidate_id"],
                job_id=test_case["job_id"],
                role_id=test_case["role_id"],
                candidate_answers=test_case.get("candidate_answers", []),
                questions=test_case.get("questions"),
            )
            results.append(result)

        # Calculate aggregates
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        avg_questions = sum(r.questions_asked for r in results) / len(results) if results else 0
        avg_answered = sum(r.questions_answered for r in results) / len(results) if results else 0
        avg_reprompts = sum(r.reprompts_given for r in results) / len(results) if results else 0
        avg_duration = (
            sum(r.call_duration_seconds for r in results) / len(results) if results else 0
        )

        batch_result = SimulationBatchResult(
            batch_id=batch_id,
            total_calls=len(results),
            successful_calls=successful,
            failed_calls=failed,
            average_questions_asked=avg_questions,
            average_questions_answered=avg_answered,
            average_reprompts=avg_reprompts,
            average_call_duration=avg_duration,
            results=results,
        )

        logger.info(
            "Batch simulation completed: batch=%s success=%d failed=%d",
            batch_id,
            successful,
            failed,
        )

        return batch_result

    def _get_default_questions(self, role_id: str) -> List[Dict[str, Any]]:
        """Get default screening questions for a role."""
        from day26_screening_scoring_engine.question_bank_loader import questions_for_role

        questions = questions_for_role(role_id)
        return [
            {
                "question_id": q.question_id,
                "category": q.category,
                "resolved_text": q.text,
                "expected_answer_type": q.expected_answer_type,
                "mandatory": q.mandatory,
                "scoring_weight": q.scoring_weight,
                "validation": {"rule": "any"},
                "routing": {"on_answer": "next", "on_no_response": "skip"},
                "state": "pending",
            }
            for q in questions
        ]

    def _score_session(
        self,
        candidate_id: str,
        job_id: str,
        session_id: str,
        role_id: str,
        answers: Dict[str, str],
        questions: List[Dict[str, Any]],
    ) -> Optional[SessionScore]:
        """Score a completed screening session."""
        try:
            # Convert answers to format expected by scoring engine
            answer_records = []
            for q in questions:
                qid = q["question_id"]
                raw_answer = answers.get(qid, "")
                answer_records.append(
                    (
                        qid,
                        {
                            "raw_answer": raw_answer,
                            "expected_answer_type": q.get("expected_answer_type", "text"),
                            "category": q.get("category", "general"),
                            "scoring_weight": q.get("scoring_weight", 1),
                            "is_mandatory": q.get("mandatory", False),
                            "intent_label": "answer" if raw_answer else "no_response",
                            "completeness": 0.8 if raw_answer else 0.0,
                            "warnings": [],
                        },
                    )
                )

            session_score = self.scoring_engine.score_session(
                candidate_id=candidate_id,
                job_id=job_id,
                session_id=session_id,
                role_id=role_id,
                answers=answer_records,
            )
            return session_score
        except Exception as e:
            logger.error("Scoring failed: error=%s", str(e))
            return None

    def _detect_intent(self, answer: str) -> str:
        """Detect intent from answer text (simplified)."""
        answer_lower = answer.lower().strip()
        if not answer_lower:
            return "no_response"
        if answer_lower in ("yes", "no", "sure", "definitely"):
            return "boolean"
        if any(c.isdigit() for c in answer_lower):
            return "numeric"
        return "answer"

    def _create_followup(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """Create a follow-up question for clarification."""
        return {
            "question_id": f"followup_{question['question_id']}",
            "category": question.get("category", "general"),
            "resolved_text": f"Could you elaborate on that regarding {question.get('resolved_text', 'this')}?",
            "expected_answer_type": "text",
            "mandatory": False,
            "scoring_weight": 1,
            "validation": {"rule": "any"},
            "routing": {"on_answer": "next", "on_no_response": "skip"},
            "state": "pending",
        }

    def get_all_results(self) -> List[SimulatedCallResult]:
        """Get all simulated call results."""
        return self._call_results.copy()
