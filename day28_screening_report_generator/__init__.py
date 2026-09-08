"""
day28_screening_report_generator
--------------------------------

Day 28 deliverable — Zecpath AI Job Portal

AI Screening Report Generator: transforms raw AI evaluations (Day 26 scoring
+ Day 27 behavioral analysis) into recruiter-friendly insights.

Core responsibilities:
  • Generate structured screening reports with key answers, strengths, risks,
    missing data.
  • Highlight: salary expectation, availability, skill confirmations.
  • Design exportable report format (JSON, HTML, printable text).
  • Provide recruiter-ready narrative and summary views.

Public API:
    from day28_screening_report_generator import ScreeningReportBuilder
    builder = ScreeningReportBuilder()
    report = builder.build_report(
        session_score=session_score_obj,
        behavioral_report=behavioral_report_obj,
        request_id="req_001",
    )
    print(report.to_recruiter_summary())
"""

from __future__ import annotations

from .export_formats import (
    EmailExporter,
    HTMLExporter,
    JSONExporter,
    PrintableExporter,
    ReportExporter,
)
from .report_builder import ScreeningReportBuilder
from .report_format import (
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
from .storage import ScreeningReportStore

__all__ = [
    "ScreeningReportBuilder",
    "ReportExporter",
    "JSONExporter",
    "HTMLExporter",
    "PrintableExporter",
    "EmailExporter",
    "ScreeningReportStore",
    "ReportFormat",
    "ReportType",
    "ScreeningReport",
    "KeyAnswer",
    "StrengthProfile",
    "RiskProfile",
    "CompensationInsights",
    "SkillConfirmation",
    "SCHEMA_VERSION",
    "MODEL_VERSION",
    "PIPELINE_VERSION",
]

__version__ = SCHEMA_VERSION
