"""
test_export_formats.py — Tests for export_formats.py
"""

from __future__ import annotations

import json

import pytest

from day28_screening_report_generator.export_formats import (
    EmailExporter,
    HTMLExporter,
    JSONExporter,
    PrintableExporter,
    get_exporter,
)
from day28_screening_report_generator.report_format import (
    CompensationInsights,
    KeyAnswer,
    RiskProfile,
    ScreeningReport,
    SkillConfirmation,
    StrengthProfile,
)


def _make_test_report() -> ScreeningReport:
    return ScreeningReport(
        candidate_id="test_cand",
        job_id="test_job",
        session_id="test_session",
        role_id="software_engineer",
        generated_at="2025-01-01T12:00:00",
        request_id="test_req",
        session_score={"normalized_score": 0.80},
        behavioral_report={
            "session": {
                "flagged_answers": 1,
                "clean_answers": 3,
                "hesitation": {"session_hesitation_rate": 0.5},
                "sentiment": {"session_sentiment_label": "positive"},
                "strength": {
                    "session_avg_strength": 0.85,
                    "session_avg_confidence": 0.88,
                    "strength_distribution": {
                        "exceptional": 1,
                        "strong": 2,
                        "competent": 1,
                        "developing": 0,
                        "weak": 0,
                    },
                    "exceptional_answers": [],
                    "weak_answers": [],
                },
                "total_contradictions": 0,
            },
            "per_answer": [
                {
                    "question_id": "Q-SE-001",
                    "category": "introduction",
                    "strength": {"overall_strength": 0.9, "strength_label": "exceptional"},
                    "red_flags": [],
                    "overall_confidence_score": 0.92,
                },
            ],
        },
        key_answers=[
            KeyAnswer(
                question_id="Q-SE-001",
                category="introduction",
                question_text="Tell me about yourself",
                candidate_answer="I am a developer",
                score=0.9,
                is_mandatory=True,
                was_answered=True,
                extracted_data={},
                red_flags=[],
                warnings=[],
            ),
        ],
        strength_profile=StrengthProfile(
            communication_strength_score=0.85,
            clarity_score=0.9,
            confidence_score=0.88,
            conviction_score=0.80,
            engagement_score=0.85,
            professionalism_score=0.9,
            key_strengths=["Clear communication"],
            development_areas=[],
        ),
        risk_profile=RiskProfile(
            hesitation_level="minimal",
            uncertainty_level="minimal",
            sentiment_risk="positive",
            contradiction_count=0,
            red_flag_count=0,
            critical_risks=[],
            warnings=[],
        ),
        compensation_insights=CompensationInsights(
            salary_expectation=20.0,
            notice_period_days=30,
            notice_period_qualifier="days",
            availability_status="within_month",
            salary_confidence="high",
            location_preferences=["Bengaluru"],
            relocation_willingness=True,
        ),
        skill_confirmation=SkillConfirmation(
            confirmed_skills=["Python"],
            skill_gaps=["SQL"],
            skill_confidence_scores={"Python": 0.9},
            experience_level="mid",
            years_experience=5.0,
        ),
        overall_red_flags=[],
        overall_warnings=[],
        recommendation="proceed",
        confidence_score=0.92,
        overall_score=0.80,
        recruiter_narrative="Strong candidate with 5 years experience.",
        executive_summary="Test candidate summary.",
        report_type="detailed_review",
    )


class TestJSONExporter:
    def test_export_returns_dict(self):
        exporter = JSONExporter()
        report = _make_test_report()
        result = exporter.export(report)
        d = json.loads(result)
        assert d["candidate_id"] == "test_cand"
        assert d["recommendation"] == "proceed"
        assert d["overall_score"] == 0.80

    def test_format_name(self):
        exporter = JSONExporter()
        assert exporter.format_name() == "json"

    def test_save_to_file(self, tmp_path):
        exporter = JSONExporter()
        report = _make_test_report()
        path = tmp_path / "report.json"
        exporter.export(report, output_path=path)
        assert path.exists()
        content = json.loads(path.read_text())
        assert content["candidate_id"] == "test_cand"


class TestHTMLExporter:
    def test_export_contains_html(self):
        exporter = HTMLExporter()
        report = _make_test_report()
        result = exporter.export(report)
        assert "<html" in result.lower()
        assert "test_cand" in result
        assert "proceed" in result

    def test_format_name(self):
        exporter = HTMLExporter()
        assert exporter.format_name() == "html"

    def test_save_to_file(self, tmp_path):
        exporter = HTMLExporter()
        report = _make_test_report()
        path = tmp_path / "report.html"
        exporter.export(report, output_path=path)
        assert path.exists()
        content = path.read_text()
        assert "<html" in content.lower()


class TestPrintableExporter:
    def test_export_contains_text(self):
        exporter = PrintableExporter()
        report = _make_test_report()
        result = exporter.export(report)
        assert "SCREENING REPORT" in result
        assert "test_cand" in result
        assert "PROCEED" in result

    def test_format_name(self):
        exporter = PrintableExporter()
        assert exporter.format_name() == "printable"

    def test_save_to_file(self, tmp_path):
        exporter = PrintableExporter()
        report = _make_test_report()
        path = tmp_path / "report.txt"
        exporter.export(report, output_path=path)
        assert path.exists()
        content = path.read_text()
        assert "SCREENING REPORT" in content


class TestEmailExporter:
    def test_export_contains_email_format(self):
        exporter = EmailExporter()
        report = _make_test_report()
        result = exporter.export(report)
        assert "Subject:" in result
        assert "SCREENING REPORT" in result
        assert "PROCEED" in result

    def test_format_name(self):
        exporter = EmailExporter()
        assert exporter.format_name() == "email"

    def test_save_to_file(self, tmp_path):
        exporter = EmailExporter()
        report = _make_test_report()
        path = tmp_path / "report.email.txt"
        exporter.export(report, output_path=path)
        assert path.exists()
        content = path.read_text()
        assert "Subject:" in content


class TestGetExporter:
    def test_get_json(self):
        exporter = get_exporter("json")
        assert isinstance(exporter, JSONExporter)

    def test_get_html(self):
        exporter = get_exporter("html")
        assert isinstance(exporter, HTMLExporter)

    def test_get_printable(self):
        exporter = get_exporter("printable")
        assert isinstance(exporter, PrintableExporter)

    def test_get_email(self):
        exporter = get_exporter("email")
        assert isinstance(exporter, EmailExporter)

    def test_unknown_format(self):
        with pytest.raises(ValueError):
            get_exporter("unknown")
