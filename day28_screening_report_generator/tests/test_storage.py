"""
test_storage.py — Tests for storage.py
"""

from __future__ import annotations

import json

import pytest

from day28_screening_report_generator.report_format import (
    CompensationInsights,
    KeyAnswer,
    RiskProfile,
    ScreeningReport,
    SkillConfirmation,
    StrengthProfile,
)
from day28_screening_report_generator.storage import ScreeningReportStore


def _make_test_report() -> ScreeningReport:
    return ScreeningReport(
        candidate_id="cand_001",
        job_id="job_01",
        session_id="sess_001",
        role_id="software_engineer",
        generated_at="2025-01-01T12:00:00",
        request_id="test_req",
        session_score={"normalized_score": 0.80},
        behavioral_report={},
        key_answers=[],
        strength_profile=StrengthProfile(
            communication_strength_score=0.85,
            clarity_score=0.9,
            confidence_score=0.88,
            conviction_score=0.80,
            engagement_score=0.85,
            professionalism_score=0.9,
            key_strengths=[],
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
        confidence_score=0.88,
        overall_score=0.80,
        recruiter_narrative="Strong candidate.",
        executive_summary="Good candidate.",
        report_type="detailed_review",
    )


class TestScreeningReportStore:
    def test_save_json(self, tmp_path):
        store = ScreeningReportStore(output_dir=tmp_path)
        report = _make_test_report()
        path = store.save_json(report)
        assert path.exists()
        content = json.loads(path.read_text())
        assert content["candidate_id"] == "cand_001"

    def test_save_report_json(self, tmp_path):
        store = ScreeningReportStore(output_dir=tmp_path)
        report = _make_test_report()
        paths = store.save_report(report, formats=["json"])
        assert "json" in paths
        assert paths["json"].exists()

    def test_save_report_html(self, tmp_path):
        store = ScreeningReportStore(output_dir=tmp_path)
        report = _make_test_report()
        paths = store.save_report(report, formats=["html"])
        assert "html" in paths
        content = paths["html"].read_text()
        assert "<html" in content.lower()

    def test_save_report_printable(self, tmp_path):
        store = ScreeningReportStore(output_dir=tmp_path)
        report = _make_test_report()
        paths = store.save_report(report, formats=["printable"])
        assert "printable" in paths
        content = paths["printable"].read_text()
        assert "SCREENING REPORT" in content

    def test_save_report_email(self, tmp_path):
        store = ScreeningReportStore(output_dir=tmp_path)
        report = _make_test_report()
        paths = store.save_report(report, formats=["email"])
        assert "email" in paths
        content = paths["email"].read_text()
        assert "Subject:" in content

    def test_save_multiple_formats(self, tmp_path):
        store = ScreeningReportStore(output_dir=tmp_path)
        report = _make_test_report()
        paths = store.save_report(report, formats=["json", "html", "printable", "email"])
        assert len(paths) == 4
        for path in paths.values():
            assert path.exists()

    def test_now_iso(self):
        from datetime import datetime

        iso = ScreeningReportStore.now_iso()
        assert isinstance(iso, str)
        assert "T" in iso

    def test_new_request_id(self):
        rid = ScreeningReportStore.new_request_id()
        assert isinstance(rid, str)
        assert len(rid) > 0
