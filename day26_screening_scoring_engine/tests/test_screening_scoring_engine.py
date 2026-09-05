"""
test_screening_scoring_engine.py
------------------------------
Unit and integration tests for ScreeningScoringEngine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from day26_screening_scoring_engine.aggregator import (
    ScoringConfig,
    classify_recommendation,
    compute_question_score,
)
from day26_screening_scoring_engine.question_bank_loader import (
    QUESTION_BANK,
    find_question,
    questions_for_role,
)
from day26_screening_scoring_engine.scoring_engine import ScreenAnswerScore, ScreeningScoringEngine

# ---- aggregator tests ---- #


class TestRecommendClassifier:
    def test_proceed_above_threshold(self):
        assert classify_recommendation(0.80) == "proceed"

    def test_proceed_at_threshold(self):
        assert classify_recommendation(0.75) == "proceed"

    def test_hold_in_range(self):
        assert classify_recommendation(0.60) == "hold"

    def test_hold_at_threshold(self):
        assert classify_recommendation(0.45) == "hold"

    def test_reject_below_threshold(self):
        assert classify_recommendation(0.30) == "reject"

    def test_reject_zero(self):
        assert classify_recommendation(0.0) == "reject"


class TestQuestionScoreBlender:
    def test_equal_weights_blend(self):
        cfg = ScoringConfig.with_overrides(
            dimension_weights={
                "clarity": 0.25,
                "relevance": 0.25,
                "completeness": 0.25,
                "consistency": 0.25,
            }
        )
        scores = {"clarity": 1.0, "relevance": 0.8, "completeness": 0.6, "consistency": 1.0}
        result = compute_question_score(scores, cfg.dimension_weights)
        expected = (1.0 * 0.25) + (0.8 * 0.25) + (0.6 * 0.25) + (1.0 * 0.25)
        assert abs(result - expected) < 1e-6

    def test_missing_dimension_defaults_to_zero(self):
        cfg = ScoringConfig.with_overrides()
        scores = {"clarity": 0.5, "relevance": 0.5}  # missing completeness, consistency
        result = compute_question_score(scores, cfg.dimension_weights)
        # The blender does NOT renormalize; missing dims simply don't contribute.
        # = 0.5 * 0.15 + 0.5 * 0.40 = 0.275
        assert abs(result - 0.275) < 1e-6

    def test_all_zeros_returns_zero(self):
        cfg = ScoringConfig.with_overrides()
        result = compute_question_score(
            {"clarity": 0.0, "relevance": 0.0, "completeness": 0.0, "consistency": 0.0},
            cfg.dimension_weights,
        )
        assert result == 0.0

    def test_all_ones_returns_one(self):
        cfg = ScoringConfig.with_overrides()
        result = compute_question_score(
            {"clarity": 1.0, "relevance": 1.0, "completeness": 1.0, "consistency": 1.0},
            cfg.dimension_weights,
        )
        assert result == 1.0


# ---- question bank tests ---- #


class TestQuestionBank:
    def test_bank_loads(self):
        assert len(QUESTION_BANK.questions) > 0
        assert len(QUESTION_BANK.roles) > 0

    def test_find_question_returns_question(self):
        q = find_question("Q-SE-001")
        assert q is not None
        assert q.scoring_weight >= 1
        assert q.category == "introduction"

    def test_find_question_unknown_returns_none(self):
        assert find_question("Q-NONE-999") is None

    def test_questions_for_role(self):
        qs = questions_for_role("software_engineer")
        assert len(qs) > 0
        # All should belong to software_engineer.
        for q in qs:
            assert q.role_id == "software_engineer"

    def test_roles_list(self):
        assert "software_engineer" in QUESTION_BANK.roles
        assert "sales_executive" in QUESTION_BANK.roles


# ---- engine tests ---- #


class TestScreeningScoringEngine:
    @pytest.fixture
    def engine(self):
        return ScreeningScoringEngine()

    def test_score_question_returns_score_object(self, engine):
        record = {
            "raw_answer": "I have 5 years of experience in Python.",
            "expected_answer_type": "text",
            "category": "experience",
            "scoring_weight": 4,
            "is_mandatory": True,
            "extracted": {"years_experience": 5.0},
            "intent_label": "answer",
            "completeness": 0.9,
            "warnings": [],
        }
        result = engine.score_question("Q-SE-005", record)
        assert isinstance(result, ScreenAnswerScore)
        assert result.question_id == "Q-SE-005"
        assert 0.0 <= result.overall_score <= 1.0
        assert 0.0 <= result.weighted_contribution <= 1.0
        assert len(result.dimension_scores) == 4
        dim_names = {d.name for d in result.dimension_scores}
        assert dim_names == {"clarity", "relevance", "completeness", "consistency"}

    def test_score_question_boolean(self, engine):
        record = {
            "raw_answer": "Yes, I am willing to relocate.",
            "expected_answer_type": "boolean",
            "category": "location",
            "scoring_weight": 4,
            "is_mandatory": True,
            "extracted": {"boolean": True},
            "intent_label": "answer",
            "completeness": 1.0,
            "warnings": [],
        }
        result = engine.score_question("Q-SE-014", record)
        assert result.overall_score >= 0.7
        assert result.was_answered is True

    def test_score_question_no_response(self, engine):
        record = {
            "raw_answer": "",
            "expected_answer_type": "text",
            "category": "experience",
            "scoring_weight": 3,
            "is_mandatory": True,
            "extracted": {},
            "intent_label": "no_response",
            "completeness": 0.0,
            "warnings": ["No response received."],
        }
        result = engine.score_question("Q-SE-007", record)
        assert result.overall_score < 0.2
        assert result.was_answered is False

    def test_score_question_off_topic(self, engine):
        record = {
            "raw_answer": "I love cricket and football.",
            "expected_answer_type": "text",
            "category": "experience",
            "scoring_weight": 2,
            "is_mandatory": False,
            "extracted": {},
            "intent_label": "off_topic",
            "completeness": 0.0,
            "warnings": [],
        }
        result = engine.score_question("Q-SE-007", record)
        assert result.overall_score == 0.0

    def test_score_question_with_objection(self, engine):
        record = {
            "raw_answer": "I don't want to answer that.",
            "expected_answer_type": "text",
            "category": "salary",
            "scoring_weight": 3,
            "is_mandatory": True,
            "extracted": {},
            "intent_label": "objection",
            "completeness": 0.2,
            "warnings": [],
        }
        result = engine.score_question("Q-SE-017", record)
        assert result.was_answered is True
        assert result.overall_score < 0.5
        assert any("objection" in w for w in result.warnings)

    def test_score_all_batch(self, engine):
        answers = [
            (
                "Q-SE-001",
                {
                    "raw_answer": "Yes, I can talk.",
                    "expected_answer_type": "boolean",
                    "category": "introduction",
                    "scoring_weight": 1,
                    "is_mandatory": True,
                    "extracted": {"boolean": True},
                    "intent_label": "answer",
                    "completeness": 1.0,
                    "warnings": [],
                },
            ),
            (
                "Q-SE-002",
                {
                    "raw_answer": "My name is Arjun and I applied for the Software Engineer role.",
                    "expected_answer_type": "text",
                    "category": "introduction",
                    "scoring_weight": 1,
                    "is_mandatory": True,
                    "extracted": {},
                    "intent_label": "answer",
                    "completeness": 1.0,
                    "warnings": [],
                },
            ),
        ]
        results = engine.score_all(answers)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, ScreenAnswerScore)
            assert 0.0 <= r.overall_score <= 1.0


# ---- integration tests ---- #


class TestSessionScoring:
    @pytest.fixture
    def engine(self):
        return ScreeningScoringEngine()

    @pytest.fixture
    def good_answers(self):
        return [
            (
                "Q-SE-001",
                {
                    "raw_answer": "Yes, this is a good time.",
                    "expected_answer_type": "boolean",
                    "category": "introduction",
                    "scoring_weight": 1,
                    "is_mandatory": True,
                    "extracted": {"boolean": True},
                    "intent_label": "answer",
                    "completeness": 1.0,
                    "warnings": [],
                },
            ),
            (
                "Q-SE-005",
                {
                    "raw_answer": "I have 5 years of professional experience.",
                    "expected_answer_type": "number",
                    "category": "experience",
                    "scoring_weight": 4,
                    "is_mandatory": True,
                    "extracted": {"years_experience": 5.0},
                    "intent_label": "answer",
                    "completeness": 0.9,
                    "warnings": [],
                },
            ),
            (
                "Q-SE-014",
                {
                    "raw_answer": "Yes, I am willing to relocate.",
                    "expected_answer_type": "boolean",
                    "category": "location",
                    "scoring_weight": 4,
                    "is_mandatory": True,
                    "extracted": {"boolean": True},
                    "intent_label": "answer",
                    "completeness": 1.0,
                    "warnings": [],
                },
            ),
        ]

    def test_session_score_produces_nonzero_score(self, engine, good_answers):
        session = engine.score_session(
            candidate_id="cand_001",
            job_id="job_backend_01",
            session_id="sess_001",
            role_id="software_engineer",
            answers=good_answers,
        )
        assert session.normalized_score > 0.0
        assert session.total_questions == 3
        assert session.answered_questions == 3
        assert session.candidate_id == "cand_001"
        assert session.session_id == "sess_001"
        assert len(session.breakdown) == 3

    def test_session_recommendation_proceed_for_good_answers(self, engine, good_answers):
        session = engine.score_session(
            candidate_id="cand_001",
            job_id="job_backend_01",
            session_id="sess_001",
            role_id="software_engineer",
            answers=good_answers,
        )
        assert session.recommendation in ("proceed", "hold")

    def test_empty_session_insufficient_data(self, engine):
        session = engine.score_session(
            candidate_id="cand_002",
            job_id="job_backend_01",
            session_id="sess_002",
            role_id="software_engineer",
            answers=[],
        )
        assert session.recommendation == "insufficient_data"
        assert session.normalized_score == 0.0
        assert session.total_questions == 0

    def test_mandatory_unanswered_triggers_reject(self, engine):
        # Two questions answered + one mandatory question unanswered.
        # This produces mandatory_unanswered=1, answered_questions=2.
        # With answered_questions (2) < min_answered (3) and mandatory_unanswered>0,
        # recommendation should be "reject" (mandatory takes precedence over insufficient_data).
        answers = [
            (
                "Q-SE-001",
                {
                    "raw_answer": "Yes, I can talk.",
                    "expected_answer_type": "boolean",
                    "category": "introduction",
                    "scoring_weight": 1,
                    "is_mandatory": True,
                    "extracted": {"boolean": True},
                    "intent_label": "answer",
                    "completeness": 1.0,
                    "warnings": [],
                },
            ),
            (
                "Q-SE-005",
                {
                    "raw_answer": "I have 5 years of experience.",
                    "expected_answer_type": "number",
                    "category": "experience",
                    "scoring_weight": 4,
                    "is_mandatory": True,
                    "extracted": {"years_experience": 5.0},
                    "intent_label": "answer",
                    "completeness": 1.0,
                    "warnings": [],
                },
            ),
            (
                "Q-SE-004",
                {
                    "raw_answer": "",
                    "expected_answer_type": "boolean",
                    "category": "education",
                    "scoring_weight": 5,
                    "is_mandatory": True,
                    "extracted": {},
                    "intent_label": "no_response",
                    "completeness": 0.0,
                    "warnings": [],
                },
            ),
        ]
        session = engine.score_session(
            candidate_id="cand_003",
            job_id="job_backend_01",
            session_id="sess_003",
            role_id="software_engineer",
            answers=answers,
        )
        assert session.mandatory_unanswered == 1
        assert session.answered_questions == 2
        assert session.recommendation == "reject"

    def test_off_topic_answer_zeroes_score(self, engine):
        answers = [
            (
                "Q-SE-005",
                {
                    "raw_answer": "My favourite cricket team is Mumbai Indians.",
                    "expected_answer_type": "number",
                    "category": "experience",
                    "scoring_weight": 4,
                    "is_mandatory": True,
                    "extracted": {},
                    "intent_label": "off_topic",
                    "completeness": 0.0,
                    "warnings": [],
                },
            ),
        ]
        session = engine.score_session(
            candidate_id="cand_004",
            job_id="job_backend_01",
            session_id="sess_004",
            role_id="software_engineer",
            answers=answers,
        )
        # The question was answered (raw text present) but is off-topic.
        assert session.breakdown[0].overall_score == 0.0
        assert session.normalized_score < 0.3

    def test_session_has_dimension_averages(self, engine, good_answers):
        session = engine.score_session(
            candidate_id="cand_001",
            job_id="job_backend_01",
            session_id="sess_001",
            role_id="software_engineer",
            answers=good_answers,
        )
        assert isinstance(session.overall_dimension_scores, dict)
        for dim in ["clarity", "relevance", "completeness", "consistency"]:
            assert dim in session.overall_dimension_scores

    def test_session_has_narrative(self, engine, good_answers):
        session = engine.score_session(
            candidate_id="cand_001",
            job_id="job_backend_01",
            session_id="sess_001",
            role_id="software_engineer",
            answers=good_answers,
        )
        assert isinstance(session.narrative, str)
        assert len(session.narrative) > 10

    def test_session_to_dict_roundtrip(self, engine, good_answers):
        session = engine.score_session(
            candidate_id="cand_001",
            job_id="job_backend_01",
            session_id="sess_001",
            role_id="software_engineer",
            answers=good_answers,
        )
        d = session.to_dict()
        assert "normalized_score" in d
        assert "breakdown" in d
        assert "narrative" in d
        assert isinstance(d["breakdown"], list)
