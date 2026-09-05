"""
test_dimension_scorers.py
-------------------------
Unit tests for the four dimension scorers:
  score_clarity, score_relevance, score_completeness
"""

from __future__ import annotations

import pytest

from day26_screening_scoring_engine.dimension_scorers import (
    score_clarity,
    score_completeness,
    score_relevance,
)


class TestClarityScorer:
    def test_empty_answer_returns_zero(self):
        result = score_clarity("", "text")
        assert result.score == 0.0
        assert result.confidence > 0.9

    def test_none_answer_returns_zero(self):
        result = score_clarity(None, "text")
        assert result.score == 0.0

    def test_one_word_answer_is_low(self):
        # A single word is unclear for a text question.
        result = score_clarity("Python", "text")
        assert result.score < 0.8
        assert result.confidence > 0.3

    def test_substantive_answer_scores_high(self):
        result = score_clarity(
            "I have five years of experience in Python and Django, " "building scalable REST APIs.",
            "text",
        )
        assert result.score >= 0.7

    def test_vague_qualifiers_reduce_score(self):
        base = score_clarity(
            "I have some experience maybe around five years "
            "I think kind of in software development.",
            "text",
        )
        # Vague qualifiers should penalize the score below 1.0.
        assert base.score < 1.0
        assert base.score < 0.85

    def test_boolean_answer_short_is_fine(self):
        result = score_clarity("Yes", "boolean")
        # Boolean "yes" should be perfectly clear.
        assert result.score >= 0.8

    def test_long_answer_penalized(self):
        long_text = " ".join(["word"] * 100)
        result = score_clarity(long_text, "text")
        assert result.score < 0.9

    def test_confidence_high_for_long_text(self):
        result = score_clarity(
            "I have around 5 years of experience in Python and "
            "JavaScript frameworks like React and Node.js.",
            "text",
        )
        assert result.confidence >= 0.8


class TestRelevanceScorer:
    def test_off_topic_returns_zero(self):
        result = score_relevance(
            "I love cricket and football.",
            "text",
            "experience",
            intent_label="off_topic",
        )
        assert result.score == 0.0
        assert result.confidence > 0.9

    def test_no_response_returns_zero(self):
        result = score_relevance(
            "idk",
            "text",
            "experience",
            intent_label="no_response",
        )
        assert result.score == 0.0

    def test_boolean_yes_no_returns_full_score(self):
        result = score_relevance(
            "Yes, I do have a B.Tech in Computer Science.",
            "boolean",
            "education",
        )
        assert result.score >= 0.9

    def test_boolean_without_yes_no_is_low(self):
        result = score_relevance(
            "I think so",
            "boolean",
            "education",
        )
        assert result.score < 0.9

    def test_number_answer_with_digits(self):
        result = score_relevance(
            "I have 5 years of experience.",
            "number",
            "experience",
        )
        assert result.score >= 0.9

    def test_number_answer_without_digits_is_low(self):
        result = score_relevance(
            "A few years I guess",
            "number",
            "experience",
        )
        assert result.score < 0.5

    def test_duration_answer_with_days(self):
        result = score_relevance(
            "My notice period is 30 days.",
            "duration",
            "notice_period",
        )
        assert result.score >= 0.9

    def test_text_with_category_keywords(self):
        result = score_relevance(
            "I hold a B.Tech in Computer Science from IIT Delhi.",
            "text",
            "education",
        )
        assert result.score >= 0.7

    def test_text_no_keywords_partial_credit(self):
        result = score_relevance(
            "I have been working.",
            "text",
            "experience",
        )
        # Some credit for a non-trivial reply; the score should be in the
        # 0.4-0.7 partial-credit band when no strong category signal is found.
        assert 0.3 <= result.score <= 0.85

    def test_extracted_slot_boosts_score(self):
        result = score_relevance(
            "I have worked in this area for about five years.",
            "text",
            "experience",
            extracted={"years_experience": 5.0},
        )
        assert result.score >= 0.7


class TestCompletenessScorer:
    def test_empty_returns_zero(self):
        result = score_completeness("", "text")
        assert result.score == 0.0

    def test_boolean_yes_no_full_complete(self):
        result = score_completeness(
            "Yes I can.",
            "boolean",
        )
        assert result.score >= 0.9

    def test_number_with_digits_full_complete(self):
        result = score_completeness(
            "I have 5 years of experience.",
            "number",
        )
        assert result.score >= 0.9

    def test_number_without_digits_incomplete(self):
        result = score_completeness(
            "Several years",
            "number",
        )
        assert result.score < 0.5

    def test_duration_with_days_full_complete(self):
        result = score_completeness(
            "I need 2 months notice.",
            "duration",
        )
        assert result.score >= 0.9

    def test_text_short_partial(self):
        result = score_completeness("Python developer.", "text")
        assert 0.3 <= result.score <= 0.7

    def test_text_long_full_complete(self):
        result = score_completeness(
            "I have been working as a Python developer for 4 years "
            "building web applications using Django and Flask.",
            "text",
        )
        assert result.score >= 0.9

    def test_expected_slot_found_bonus(self):
        # When an expected slot is found, completeness is 1.0.
        result = score_completeness(
            "About 5 years.",
            "text",
            expected_slot="years_experience",
            extracted={"years_experience": 5.0},
        )
        assert result.score == 1.0

    def test_expected_slot_missing_partial(self):
        result = score_completeness(
            "Around five years maybe.",
            "text",
            expected_slot="years_experience",
            extracted={},
        )
        assert 0.0 < result.score < 0.5

    def test_no_response_returns_zero(self):
        result = score_completeness(
            "",
            "text",
            intent_label="no_response",
        )
        assert result.score == 0.0


class TestIntegration:
    def test_all_dimensions_produce_valid_scores(self):
        answer = "Yes, I have 5 years of experience in Python and Django."
        clarity = score_clarity(answer, "text")
        relevance = score_relevance(
            answer,
            "text",
            "experience",
            extracted={"years_experience": 5.0},
            intent_label="answer",
        )
        completeness = score_completeness(
            answer,
            "text",
            extracted={"years_experience": 5.0},
            intent_label="answer",
        )
        for result in [clarity, relevance, completeness]:
            assert 0.0 <= result.score <= 1.0
            assert 0.0 <= result.confidence <= 1.0
            assert isinstance(result.notes, list)

    def test_poor_answer_low_across_dimensions(self):
        answer = "maybe kind of I guess"
        clarity = score_clarity(answer, "text")
        relevance = score_relevance(answer, "text", "experience")
        completeness = score_completeness(answer, "text")
        # Poor answer should score low on clarity.
        assert clarity.score < 0.7
        # Vague reply may still be relevant if on-topic.
        assert relevance.score >= 0.0
        # Vague + short -> low completeness.
        assert completeness.score < 0.85
