"""
test_report_builder.py — Tests for report_builder.py
"""

from __future__ import annotations

import pytest

from day28_screening_report_generator.report_builder import ScreeningReportBuilder


class TestScreeningReportBuilder:
    @pytest.fixture
    def builder(self):
        return ScreeningReportBuilder()

    @pytest.fixture
    def session_score(self):
        return {
            "candidate_id": "cand_001",
            "job_id": "job_01",
            "session_id": "sess_001",
            "role_id": "software_engineer",
            "total_questions": 8,
            "answered_questions": 8,
            "mandatory_unanswered": 0,
            "normalized_score": 0.78,
            "overall_dimension_scores": {
                "clarity": 0.85,
                "relevance": 0.78,
                "completeness": 0.75,
                "consistency": 0.82,
            },
            "hard_filters_failed": [],
            "recommendation": "proceed",
            "consistency": {"score": 0.90, "is_consistent": True, "flagged_questions": []},
            "breakdown": [
                {
                    "question_id": "Q-SE-001",
                    "category": "introduction",
                    "scoring_weight": 1,
                    "is_mandatory": True,
                    "was_answered": True,
                    "overall_score": 0.90,
                    "dimension_scores": {"clarity": 0.95, "relevance": 0.90, "completeness": 0.85},
                    "explanation": "Strong self introduction.",
                },
                {
                    "question_id": "Q-SE-005",
                    "category": "experience",
                    "scoring_weight": 4,
                    "is_mandatory": True,
                    "was_answered": True,
                    "overall_score": 0.75,
                    "dimension_scores": {"clarity": 0.80, "relevance": 0.75, "completeness": 0.70},
                    "explanation": "5 years experience.",
                },
                {
                    "question_id": "Q-SE-017",
                    "category": "salary",
                    "scoring_weight": 4,
                    "is_mandatory": True,
                    "was_answered": True,
                    "overall_score": 0.80,
                    "dimension_scores": {"clarity": 0.85, "relevance": 0.80, "completeness": 0.75},
                    "explanation": "Salary 18-20 LPA.",
                },
                {
                    "question_id": "Q-SE-019",
                    "category": "notice_period",
                    "scoring_weight": 4,
                    "is_mandatory": True,
                    "was_answered": True,
                    "overall_score": 0.85,
                    "dimension_scores": {"clarity": 0.90, "relevance": 0.85, "completeness": 0.80},
                    "explanation": "Notice period 30 days.",
                },
            ],
            "narrative": "Candidate answered all 8 questions with proceed recommendation.",
            "warnings": [],
        }

    @pytest.fixture
    def behavioral_report(self):
        return {
            "candidate_id": "cand_001",
            "job_id": "job_01",
            "session_id": "sess_001",
            "role_id": "software_engineer",
            "session": {
                "total_answers": 4,
                "flagged_answers": 0,
                "clean_answers": 4,
                "hesitation": {
                    "session_hesitation_rate": 0.5,
                    "session_avg_hesitation_severity": 0.2,
                    "high_hesitation_answers": [],
                },
                "pace": {
                    "avg_response_length": 35.0,
                    "avg_pace_wps": 2.2,
                    "avg_filler_ratio": 0.014,
                    "pace_distribution": {"normal": 4},
                    "length_distribution": {"normal": 4},
                },
                "sentiment": {
                    "session_sentiment_polarity": 0.35,
                    "session_sentiment_label": "positive",
                    "sentiment_distribution": {"positive": 3, "neutral": 1},
                    "avg_positive_score": 0.60,
                    "avg_negative_score": 0.05,
                    "avg_neutral_score": 0.35,
                    "negative_sentiment_answers": [],
                    "mixed_sentiment_answers": [],
                },
                "uncertainty": {
                    "session_uncertainty_rate": 0.25,
                    "high_uncertainty_answers": [],
                },
                "strength": {
                    "session_avg_strength": 0.82,
                    "session_avg_confidence": 0.85,
                    "strength_distribution": {
                        "exceptional": 1,
                        "strong": 2,
                        "competent": 1,
                        "developing": 0,
                        "weak": 0,
                    },
                    "exceptional_answers": ["Q-SE-001"],
                    "weak_answers": [],
                },
                "total_contradictions": 0,
                "internal_contradictions": 0,
                "cross_answer_contradictions": 0,
                "severe_contradictions": [],
            },
            "per_answer": [
                {
                    "question_id": "Q-SE-001",
                    "category": "introduction",
                    "raw_answer": "Hello, I'm a software engineer.",
                    "hesitation": {
                        "total_patterns": 0,
                        "filler_count": 0,
                        "pause_count": 0,
                        "repetition_count": 0,
                        "repair_count": 0,
                        "false_start_count": 0,
                        "max_severity": 0.0,
                        "avg_severity": 0.0,
                        "severity_label": "minimal",
                        "flagged": False,
                    },
                    "pace": {
                        "word_count": 8,
                        "char_count": 40,
                        "sentence_count": 1,
                        "avg_sentence_length": 8.0,
                        "duration_seconds": 3.0,
                        "words_per_second": 2.67,
                        "pace_label": "normal",
                        "pace_score": 1.0,
                        "length_label": "normal",
                        "length_score": 1.0,
                        "filler_ratio": 0.0,
                        "flagged": False,
                    },
                    "sentiment": {
                        "polarity": 0.50,
                        "sentiment_label": "positive",
                        "positive_score": 0.70,
                        "negative_score": 0.05,
                        "neutral_score": 0.25,
                        "confidence": 0.90,
                        "emotion_signals": {"happy": 0.70},
                        "dominant_emotion": "happy",
                        "flagged": False,
                    },
                    "uncertainty": {
                        "total_signals": 0,
                        "hedge_count": 0,
                        "doubt_count": 0,
                        "vague_quantifier_count": 0,
                        "max_severity": 0.0,
                        "avg_severity": 0.0,
                        "severity_label": "minimal",
                        "flagged": False,
                    },
                    "strength": {
                        "clarity": 0.95,
                        "confidence": 0.90,
                        "conviction": 0.85,
                        "engagement": 0.80,
                        "professionalism": 0.90,
                        "overall_strength": 0.88,
                        "strength_label": "exceptional",
                        "primary_strength": "clarity",
                        "primary_weakness": "engagement",
                    },
                    "overall_confidence_score": 0.92,
                    "red_flags": [],
                    "warnings": [],
                },
                {
                    "question_id": "Q-SE-005",
                    "category": "experience",
                    "raw_answer": "I have 5 years of experience in Python development.",
                    "hesitation": {
                        "total_patterns": 0,
                        "filler_count": 0,
                        "pause_count": 0,
                        "repetition_count": 0,
                        "repair_count": 0,
                        "false_start_count": 0,
                        "max_severity": 0.0,
                        "avg_severity": 0.0,
                        "severity_label": "minimal",
                        "flagged": False,
                    },
                    "pace": {
                        "word_count": 12,
                        "char_count": 60,
                        "sentence_count": 1,
                        "avg_sentence_length": 12.0,
                        "duration_seconds": 4.0,
                        "words_per_second": 3.0,
                        "pace_label": "normal",
                        "pace_score": 1.0,
                        "length_label": "normal",
                        "length_score": 1.0,
                        "filler_ratio": 0.0,
                        "flagged": False,
                    },
                    "sentiment": {
                        "polarity": 0.30,
                        "sentiment_label": "positive",
                        "positive_score": 0.55,
                        "negative_score": 0.05,
                        "neutral_score": 0.40,
                        "confidence": 0.85,
                        "emotion_signals": {"neutral": 0.40},
                        "dominant_emotion": "neutral",
                        "flagged": False,
                    },
                    "uncertainty": {
                        "total_signals": 0,
                        "hedge_count": 0,
                        "doubt_count": 0,
                        "vague_quantifier_count": 0,
                        "max_severity": 0.0,
                        "avg_severity": 0.0,
                        "severity_label": "minimal",
                        "flagged": False,
                    },
                    "strength": {
                        "clarity": 0.85,
                        "confidence": 0.85,
                        "conviction": 0.80,
                        "engagement": 0.85,
                        "professionalism": 0.85,
                        "overall_strength": 0.84,
                        "strength_label": "strong",
                        "primary_strength": "clarity",
                        "primary_weakness": "engagement",
                    },
                    "overall_confidence_score": 0.85,
                    "red_flags": [],
                    "warnings": [],
                },
            ],
            "total_answers": 4,
            "red_flag_count": 0,
            "warning_count": 0,
            "overall_confidence_score": 0.85,
            "overall_strength_score": 0.82,
            "overall_sentiment_polarity": 0.35,
            "narrative": "Strong communication throughout.",
            "warnings": [],
        }

    def test_build_report(self, builder, session_score, behavioral_report):
        report = builder.build_report(
            session_score=session_score,
            behavioral_report=behavioral_report,
            request_id="test_req_001",
        )

        assert report.candidate_id == "cand_001"
        assert report.job_id == "job_01"
        assert report.session_id == "sess_001"
        assert report.role_id == "software_engineer"
        assert report.recommendation == "proceed"
        assert report.overall_score == 0.78
        assert report.confidence_score > 0.7

    def test_report_has_key_components(self, builder, session_score, behavioral_report):
        report = builder.build_report(
            session_score=session_score,
            behavioral_report=behavioral_report,
        )

        # Check all major sections exist.
        assert report.key_answers is not None
        assert report.strength_profile is not None
        assert report.risk_profile is not None
        assert report.compensation_insights is not None
        assert report.skill_confirmation is not None
        assert report.overall_red_flags is not None
        assert report.overall_warnings is not None
        assert report.recruiter_narrative is not None
        assert report.executive_summary is not None

    def test_salary_extraction(self, builder, session_score, behavioral_report):
        report = builder.build_report(
            session_score=session_score,
            behavioral_report=behavioral_report,
        )

        # Salary should be extracted from Q-SE-017.
        comp = report.compensation_insights
        assert comp.salary_expectation is not None
        assert comp.salary_expectation > 0
        assert comp.salary_confidence in ("high", "medium", "low", "uncertain")

    def test_notice_period_extraction(self, builder, session_score, behavioral_report):
        report = builder.build_report(
            session_score=session_score,
            behavioral_report=behavioral_report,
        )

        comp = report.compensation_insights
        assert comp.notice_period_days == 30

    def test_skill_confirmation(self, builder, session_score, behavioral_report):
        report = builder.build_report(
            session_score=session_score,
            behavioral_report=behavioral_report,
        )

        skills = report.skill_confirmation
        assert skills.years_experience == 5.0
        assert skills.experience_level in ("entry", "mid", "senior", "lead")

    def test_risk_profile(self, builder, session_score, behavioral_report):
        report = builder.build_report(
            session_score=session_score,
            behavioral_report=behavioral_report,
        )

        risk = report.risk_profile
        assert risk.hesitation_level in ("minimal", "moderate", "high", "severe")
        assert risk.uncertainty_level in ("minimal", "moderate", "high", "severe")
        assert risk.sentiment_risk in ("positive", "neutral", "negative")

    def test_to_dict(self, builder, session_score, behavioral_report):
        report = builder.build_report(
            session_score=session_score,
            behavioral_report=behavioral_report,
        )

        d = report.to_dict()
        assert d["candidate_id"] == "cand_001"
        assert "key_answers" in d
        assert "strength_profile" in d
        assert "risk_profile" in d
        assert "compensation_insights" in d
        assert "skill_confirmation" in d

    def test_recruiter_summary(self, builder, session_score, behavioral_report):
        report = builder.build_report(
            session_score=session_score,
            behavioral_report=behavioral_report,
        )

        summary = report.to_recruiter_summary()
        assert "cand_001" in summary
        assert "PROCEED" in summary
        assert "78%" in summary or "78.0%" in summary


class TestExtractors:
    """Test the internal extraction helpers."""

    def test_salary_extraction(self):
        from day28_screening_report_generator.report_builder import _AnswerExtractor

        # Test various salary formats.
        assert _AnswerExtractor.extract_salary_expectation("I want 18 LPA") == 18.0
        assert _AnswerExtractor.extract_salary_expectation("Looking for 15 lakhs per annum") == 15.0
        assert _AnswerExtractor.extract_salary_expectation("Around 20 to 25 LPA") == 22.5
        assert (
            _AnswerExtractor.extract_salary_expectation("100k per year") == 100000.0
        )  # 100k in thousands
        assert _AnswerExtractor.extract_salary_expectation("No salary mentioned") is None

    def test_notice_period_extraction(self):
        from day28_screening_report_generator.report_builder import _AnswerExtractor

        assert _AnswerExtractor.extract_notice_period("30 days notice") == 30
        assert _AnswerExtractor.extract_notice_period("Notice period is 60 days") == 60
        assert _AnswerExtractor.extract_notice_period("I can join in 45 days") == 45
        assert _AnswerExtractor.extract_notice_period("No notice") is None

    def test_experience_extraction(self):
        from day28_screening_report_generator.report_builder import _AnswerExtractor

        assert _AnswerExtractor.extract_experience("5 years experience") == 5.0
        assert _AnswerExtractor.extract_experience("3.5 years of work") == 3.5
        assert _AnswerExtractor.extract_experience("I have 10 years experience") == 10.0
        assert _AnswerExtractor.extract_experience("No experience mentioned") is None

    def test_location_extraction(self):
        from day28_screening_report_generator.report_builder import _AnswerExtractor

        locs = _AnswerExtractor.extract_location("I'm based in Bengaluru")
        assert "Bengaluru" in locs

        locs = _AnswerExtractor.extract_location("Can work from Mumbai or Bangalore")
        assert len(locs) >= 1

    def test_relocation_willingness(self):
        from day28_screening_report_generator.report_builder import _AnswerExtractor

        assert _AnswerExtractor.extract_relocation_willingness("Yes, willing to relocate") is True
        assert _AnswerExtractor.extract_relocation_willingness("Not willing to relocate") is False
        assert _AnswerExtractor.extract_relocation_willingness("Maybe") is None

    def test_availability_extraction(self):
        from day28_screening_report_generator.report_builder import _AnswerExtractor

        assert _AnswerExtractor.extract_availability("Immediate joining") == "immediate"
        assert _AnswerExtractor.extract_availability("Can join within a month") == "within_month"
        assert _AnswerExtractor.extract_availability("Available in 2 months") == "within_2_months"
        assert _AnswerExtractor.extract_availability("Not sure") == "uncertain"
        assert _AnswerExtractor.extract_availability("") == "uncertain"
