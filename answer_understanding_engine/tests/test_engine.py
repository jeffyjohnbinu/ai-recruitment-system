"""
answer_understanding_engine/tests/test_engine.py
-------------------------------------------------
Pytest suite for the Answer Understanding Engine.

Run with:
    pytest answer_understanding_engine/tests/test_engine.py -v
"""

from __future__ import annotations

from answer_understanding_engine.engine import AnswerUnderstandingEngine
from answer_understanding_engine.engine import _normalize_duration as normalize_duration
from answer_understanding_engine.engine import _normalize_money_to_lakhs as normalize_money_to_lakhs


class TestIntentClassification:
    """Tests for intent detection."""

    def setup_method(self):
        self.engine = AnswerUnderstandingEngine()

    def test_empty_answer_is_no_response(self):
        result = self.engine.understand("", question_id="Q-SE-001")
        assert result.intent.label == "no_response"
        assert result.is_missing

    def test_filler_answer_is_no_response(self):
        for text in ["I don't know", "idk", "not sure", "hmm", "uh", "N/A"]:
            result = self.engine.understand(text, question_id="Q-SE-001")
            assert result.intent.label == "no_response", f"Failed for: {text}"
            assert result.is_missing

    def test_question_is_clarification_request(self):
        for text in ["What do you mean?", "Can you clarify?", "Why do you ask?"]:
            result = self.engine.understand(text, question_id="Q-SE-001")
            assert result.intent.label == "clarification_request", f"Failed for: {text}"

    def test_redirect_attempt(self):
        for text in [
            "Can we talk about salary instead?",
            "Let's skip this question.",
            "I'd rather discuss the benefits.",
        ]:
            result = self.engine.understand(text, question_id="Q-SE-001")
            assert result.intent.label == "redirect", f"Failed for: {text}"

    def test_objection(self):
        for text in [
            "I don't want to answer that.",
            "This doesn't feel right.",
            "Why should I tell you?",
        ]:
            result = self.engine.understand(text, question_id="Q-SE-001")
            assert result.intent.label == "objection", f"Failed for: {text}"

    def test_off_topic(self):
        for text in [
            "Here's a recipe for biryani.",
            "Check out my YouTube channel!",
            "The cricket score is 245/3.",
        ]:
            result = self.engine.understand(text, question_id="Q-SE-001")
            assert result.intent.label == "off_topic", f"Failed for: {text}"
            assert result.is_off_topic

    def test_normal_answer(self):
        result = self.engine.understand("Yes, I have 5 years experience.", question_id="Q-SE-001")
        assert result.intent.label == "answer"


class TestSlotExtraction:
    """Tests for slot/value extraction."""

    def setup_method(self):
        self.engine = AnswerUnderstandingEngine()

    def test_years_experience_numeric(self):
        result = self.engine.understand("I have 5 years of experience.", question_id="Q-SE-006")
        assert "years_experience" in result.extracted
        assert result.extracted["years_experience"] == 5.0

    def test_years_experience_with_plus(self):
        result = self.engine.understand("About 3+ yrs", question_id="Q-SE-006")
        assert result.extracted["years_experience"] == 3.0

    def test_years_experience_fresher(self):
        result = self.engine.understand("I'm a fresher.", question_id="Q-SE-006")
        assert result.extracted["years_experience"] == 0.0

    def test_notice_period_days(self):
        result = self.engine.understand("30 days notice", question_id="Q-SE-019")
        assert "notice_period_days" in result.extracted
        assert result.extracted["notice_period_days"] == 30

    def test_notice_period_weeks(self):
        result = self.engine.understand("2 weeks", question_id="Q-SE-019")
        assert result.extracted["notice_period_days"] == 14

    def test_notice_period_months(self):
        result = self.engine.understand("1 month", question_id="Q-SE-019")
        assert result.extracted["notice_period_days"] == 30

    def test_immediate_availability(self):
        result = self.engine.understand("I can join immediately.", question_id="Q-SE-019")
        assert result.extracted["availability"] == "immediate"

    def test_join_by_date(self):
        result = self.engine.understand("By 15/03/2026", question_id="Q-SE-021")
        assert "join_by_date" in result.extracted

    def test_salary_lpa(self):
        result = self.engine.understand("I expect 18 LPA.", question_id="Q-SE-017")
        assert "expected_salary_lakhs" in result.extracted
        assert result.extracted["expected_salary_lakhs"] == 18.0

    def test_salary_lakhs(self):
        result = self.engine.understand("15 lakhs per annum", question_id="Q-SE-017")
        assert result.extracted["expected_salary_lakhs"] == 15.0

    def test_salary_raw_inr_converted(self):
        result = self.engine.understand("₹1800000 per year", question_id="Q-SE-017")
        assert "expected_salary_lakhs" in result.extracted
        assert result.extracted["expected_salary_lakhs"] == 18.0

    def test_boolean_yes(self):
        result = self.engine.understand("Yes, I am willing.", question_id="Q-SE-014")
        assert result.extracted["boolean"] is True

    def test_boolean_no(self):
        result = self.engine.understand("No, I cannot relocate.", question_id="Q-SE-014")
        assert result.extracted["boolean"] is False

    def test_skills_extraction(self):
        result = self.engine.understand(
            "I know Python, React, AWS, and PostgreSQL.", question_id="Q-SE-009"
        )
        assert "skills" in result.extracted
        skills = result.extracted["skills"]
        assert "python" in skills
        assert "react" in skills
        assert "aws" in skills
        assert "postgresql" in skills

    def test_city_extraction(self):
        result = self.engine.understand("I'm based in Bangalore.", question_id="Q-SE-013")
        assert "city" in result.extracted
        assert result.extracted["city"] == "bengaluru"


class TestQualityFlags:
    """Tests for vague / missing / off-topic detection."""

    def setup_method(self):
        self.engine = AnswerUnderstandingEngine()

    def test_vague_answer(self):
        result = self.engine.understand("I kind of know Python, maybe.", question_id="Q-SE-009")
        assert result.is_vague
        assert result.completeness < 1.0

    def test_missing_expected_slot(self):
        result = self.engine.understand(
            "I like coding.", question_id="Q-SE-006", expected_slot="years_experience"
        )
        assert result.is_missing
        assert "years_experience" not in result.extracted

    def test_completeness_on_topic(self):
        result = self.engine.understand("5 years in Python and Django.", question_id="Q-SE-006")
        assert result.completeness == 1.0

    def test_completeness_off_topic(self):
        result = self.engine.understand("Here is my recipe.", question_id="Q-SE-006")
        assert result.completeness == 0.0


class TestNormalizationHelpers:
    """Tests for the small helper functions."""

    def test_normalize_duration_days(self):
        assert normalize_duration("30", "days") == 30

    def test_normalize_duration_weeks(self):
        assert normalize_duration("2", "weeks") == 14

    def test_normalize_duration_months(self):
        assert normalize_duration("1", "months") == 30

    def test_normalize_money_to_lakhs_small_is_lakhs(self):
        # Under 1000 treated as lakhs
        assert normalize_money_to_lakhs("18") == 18.0
        assert normalize_money_to_lakhs("15.5") == 15.5

    def test_normalize_money_to_lakhs_large_is_inr(self):
        # 1800000 -> 18 lakhs
        assert normalize_money_to_lakhs("1800000") == 18.0
        assert normalize_money_to_lakhs("1200000") == 12.0

    def test_normalize_money_none_on_bad_input(self):
        assert normalize_money_to_lakhs("abc") is None


class TestStructuredAnswerToDict:
    """Ensure the dataclass serialises correctly."""

    def test_to_dict_contains_all_fields(self):
        engine = AnswerUnderstandingEngine()
        result = engine.understand(
            "I have 3 years Python experience.",
            question_id="Q-SE-006",
        )
        d = result.to_dict()
        assert d["question_id"] == "Q-SE-006"
        assert d["intent"]["label"] == "answer"
        assert "years_experience" in d["extracted"]
        assert "slots" in d
        assert isinstance(d["warnings"], list)
