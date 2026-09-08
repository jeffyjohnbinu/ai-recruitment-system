"""
test_report_format.py — Tests for report_format dataclasses.
"""

from __future__ import annotations

import pytest

from day28_screening_report_generator.report_format import (
    MODEL_VERSION,
    PIPELINE_VERSION,
    SCHEMA_VERSION,
    CompensationInsights,
    KeyAnswer,
    ReportFormat,
    ReportType,
    RiskProfile,
    ScreeningReport,
    SkillConfirmation,
    StrengthProfile,
)

# ---- ReportFormat / ReportType constants ---- #


class TestReportFormatConstants:
    def test_json_format(self):
        assert ReportFormat.JSON == "json"

    def test_html_format(self):
        assert ReportFormat.HTML == "html"

    def test_printable_format(self):
        assert ReportFormat.PRINTABLE == "printable"

    def test_email_format(self):
        assert ReportFormat.EMAIL == "email"


class TestReportTypeConstants:
    def test_shortlisting(self):
        assert ReportType.SHORTLISTING == "shortlisting"

    def test_detailed_review(self):
        assert ReportType.DETAILED_REVIEW == "detailed_review"

    def test_comparative(self):
        assert ReportType.COMPARATIVE == "comparative"

    def test_executive_summary(self):
        assert ReportType.EXECUTIVE_SUMMARY == "executive_summary"


# ---- KeyAnswer ---- #


class TestKeyAnswer:
    def test_creation(self):
        ka = KeyAnswer(
            question_id="Q-SE-001",
            category="introduction",
            question_text="Tell me about yourself",
            candidate_answer="I am a developer",
            score=0.85,
            is_mandatory=True,
            was_answered=True,
            extracted_data={"skill": "Python"},
            red_flags=[],
            warnings=[],
        )
        assert ka.question_id == "Q-SE-001"
        assert ka.score == 0.85
        assert ka.is_mandatory is True

    def test_to_dict(self):
        ka = KeyAnswer(
            question_id="Q-SE-001",
            category="introduction",
            question_text="Test Q",
            candidate_answer="Test A",
            score=0.85,
            is_mandatory=True,
            was_answered=True,
            extracted_data={},
            red_flags=[],
            warnings=[],
        )
        d = ka.to_dict()
        assert d["question_id"] == "Q-SE-001"
        assert d["score"] == 0.85


# ---- StrengthProfile ---- #


class TestStrengthProfile:
    def test_creation(self):
        sp = StrengthProfile(
            communication_strength_score=0.85,
            clarity_score=0.90,
            confidence_score=0.80,
            conviction_score=0.75,
            engagement_score=0.85,
            professionalism_score=0.90,
            key_strengths=["Clear communication"],
            development_areas=["Some nervousness"],
        )
        assert sp.communication_strength_score == 0.85
        assert sp.clarity_score == 0.90

    def test_to_dict(self):
        sp = StrengthProfile(
            communication_strength_score=0.85,
            clarity_score=0.90,
            confidence_score=0.80,
            conviction_score=0.75,
            engagement_score=0.85,
            professionalism_score=0.90,
            key_strengths=[],
            development_areas=[],
        )
        d = sp.to_dict()
        assert d["communication_strength_score"] == 0.85
        assert d["clarity_score"] == 0.90
        assert d["key_strengths"] == []
        assert d["development_areas"] == []


# ---- RiskProfile ---- #


class TestRiskProfile:
    def test_creation(self):
        rp = RiskProfile(
            hesitation_level="moderate",
            uncertainty_level="minimal",
            sentiment_risk="positive",
            contradiction_count=0,
            red_flag_count=2,
            critical_risks=["High hesitation"],
            warnings=["Minor issue"],
        )
        assert rp.hesitation_level == "moderate"
        assert rp.uncertainty_level == "minimal"
        assert rp.red_flag_count == 2

    def test_to_dict(self):
        rp = RiskProfile(
            hesitation_level="minimal",
            uncertainty_level="minimal",
            sentiment_risk="neutral",
            contradiction_count=0,
            red_flag_count=0,
            critical_risks=[],
            warnings=[],
        )
        d = rp.to_dict()
        assert d["hesitation_level"] == "minimal"
        assert d["red_flag_count"] == 0


# ---- CompensationInsights ---- #


class TestCompensationInsights:
    def test_full(self):
        ci = CompensationInsights(
            salary_expectation=20.0,
            notice_period_days=30,
            notice_period_qualifier="days",
            availability_status="within_month",
            salary_confidence="high",
            location_preferences=["Bengaluru"],
            relocation_willingness=True,
        )
        assert ci.salary_expectation == 20.0
        assert ci.availability_status == "within_month"
        assert ci.relocation_willingness is True

    def test_empty(self):
        ci = CompensationInsights(
            salary_expectation=None,
            notice_period_days=None,
            notice_period_qualifier=None,
            availability_status="not_specified",
            salary_confidence="uncertain",
            location_preferences=[],
            relocation_willingness=None,
        )
        assert ci.salary_expectation is None
        assert ci.availability_status == "not_specified"

    def test_to_dict(self):
        ci = CompensationInsights(
            salary_expectation=None,
            notice_period_days=None,
            notice_period_qualifier=None,
            availability_status="not_specified",
            salary_confidence="uncertain",
            location_preferences=[],
            relocation_willingness=None,
        )
        d = ci.to_dict()
        assert d["salary_expectation"] is None
        assert d["availability_status"] == "not_specified"


# ---- SkillConfirmation ---- #


class TestSkillConfirmation:
    def test_creation(self):
        sc = SkillConfirmation(
            confirmed_skills=["Python", "JavaScript"],
            skill_gaps=["SQL"],
            skill_confidence_scores={"Python": 0.9, "JavaScript": 0.8},
            experience_level="mid",
            years_experience=4.5,
        )
        assert "Python" in sc.confirmed_skills
        assert "SQL" in sc.skill_gaps
        assert sc.experience_level == "mid"
        assert sc.years_experience == 4.5

    def test_to_dict(self):
        sc = SkillConfirmation(
            confirmed_skills=["Python"],
            skill_gaps=[],
            skill_confidence_scores={},
            experience_level="entry",
            years_experience=None,
        )
        d = sc.to_dict()
        assert d["confirmed_skills"] == ["Python"]
        assert d["skill_gaps"] == []
        assert d["years_experience"] is None


# ---- ScreeningReport ---- #


class TestScreeningReport:
    @pytest.fixture
    def sample_report(self):
        return ScreeningReport(
            candidate_id="cand_001",
            job_id="job_01",
            session_id="sess_001",
            role_id="software_engineer",
            generated_at="2025-01-01T00:00:00",
            request_id="req_001",
            session_score={"normalized_score": 0.75},
            behavioral_report={},
            key_answers=[],
            strength_profile=StrengthProfile(
                communication_strength_score=0.80,
                clarity_score=0.85,
                confidence_score=0.80,
                conviction_score=0.70,
                engagement_score=0.85,
                professionalism_score=0.80,
                key_strengths=[],
                development_areas=[],
            ),
            risk_profile=RiskProfile(
                hesitation_level="minimal",
                uncertainty_level="minimal",
                sentiment_risk="neutral",
                contradiction_count=0,
                red_flag_count=0,
                critical_risks=[],
                warnings=[],
            ),
            compensation_insights=CompensationInsights(
                salary_expectation=None,
                notice_period_days=None,
                notice_period_qualifier=None,
                availability_status="not_specified",
                salary_confidence="uncertain",
                location_preferences=[],
                relocation_willingness=None,
            ),
            skill_confirmation=SkillConfirmation(
                confirmed_skills=[],
                skill_gaps=[],
                skill_confidence_scores={},
                experience_level="entry",
                years_experience=None,
            ),
            overall_red_flags=[],
            overall_warnings=[],
            recommendation="proceed",
            confidence_score=0.85,
            overall_score=0.75,
            recruiter_narrative="Candidate looks good.",
            executive_summary="Strong candidate.",
            report_type="detailed_review",
            format="json",
        )

    def test_to_dict(self, sample_report):
        d = sample_report.to_dict()
        assert d["candidate_id"] == "cand_001"
        assert d["job_id"] == "job_01"
        assert d["recommendation"] == "proceed"
        assert d["overall_score"] == 0.75
        assert d["confidence_score"] == 0.85
        assert d["schema_version"] == SCHEMA_VERSION
        assert d["model_version"] == MODEL_VERSION
        assert d["pipeline_version"] == PIPELINE_VERSION

    def test_to_recruiter_summary(self, sample_report):
        summary = sample_report.to_recruiter_summary()
        assert "cand_001" in summary
        assert "75.0%" in summary or "75%" in summary
        assert "PROCEED" in summary


# ---- Version metadata ---- #


class TestVersionMetadata:
    def test_schema_version(self):
        assert SCHEMA_VERSION == "1.0.0"

    def test_model_version(self):
        assert MODEL_VERSION == "screening-report-generator-1.0.0"

    def test_pipeline_version(self):
        assert PIPELINE_VERSION == "zecpath-day28"
