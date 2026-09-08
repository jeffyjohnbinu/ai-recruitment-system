"""
report_format.py
-----------------
Dataclass definitions for the screening report format.

Combines Day 26 scoring data (SessionScore) with Day 27 behavioral insights
(BehavioralIndicatorsReport) into a single recruiter-friendly report structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "screening-report-generator-1.0.0"
PIPELINE_VERSION = "zecpath-day28"


# Report formats
class ReportFormat:
    JSON = "json"
    HTML = "html"
    PRINTABLE = "printable"
    EMAIL = "email"


class ReportType:
    SHORTLISTING = "shortlisting"
    DETAILED_REVIEW = "detailed_review"
    COMPARATIVE = "comparative"
    EXECUTIVE_SUMMARY = "executive_summary"


@dataclass
class KeyAnswer:
    """One key Q&A extracted from the raw answers."""

    question_id: str
    category: str
    question_text: str
    candidate_answer: str
    score: float
    is_mandatory: bool
    was_answered: bool
    extracted_data: Dict[str, Any]
    red_flags: List[str]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "category": self.category,
            "question_text": self.question_text,
            "candidate_answer": self.candidate_answer,
            "score": self.score,
            "is_mandatory": self.is_mandatory,
            "was_answered": self.was_answered,
            "extracted_data": self.extracted_data,
            "red_flags": self.red_flags,
            "warnings": self.warnings,
        }


@dataclass
class StrengthProfile:
    """Aggregated strengths for the candidate."""

    communication_strength_score: float  # 0-1
    clarity_score: float  # 0-1
    confidence_score: float  # 0-1
    conviction_score: float  # 0-1
    engagement_score: float  # 0-1
    professionalism_score: float  # 0-1
    key_strengths: List[str]
    development_areas: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "communication_strength_score": self.communication_strength_score,
            "clarity_score": self.clarity_score,
            "confidence_score": self.confidence_score,
            "conviction_score": self.conviction_score,
            "engagement_score": self.engagement_score,
            "professionalism_score": self.professionalism_score,
            "key_strengths": self.key_strengths,
            "development_areas": self.development_areas,
        }


@dataclass
class RiskProfile:
    """Aggregated risks and concerns for the candidate."""

    hesitation_level: str  # "minimal", "moderate", "high", "severe"
    uncertainty_level: str  # "minimal", "moderate", "high", "severe"
    sentiment_risk: str  # "positive", "neutral", "negative"
    contradiction_count: int
    red_flag_count: int
    critical_risks: List[str]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hesitation_level": self.hesitation_level,
            "uncertainty_level": self.uncertainty_level,
            "sentiment_risk": self.sentiment_risk,
            "contradiction_count": self.contradiction_count,
            "red_flag_count": self.red_flag_count,
            "critical_risks": self.critical_risks,
            "warnings": self.warnings,
        }


@dataclass
class CompensationInsights:
    """Extracted compensation and availability information."""

    salary_expectation: Optional[float]
    notice_period_days: Optional[int]
    notice_period_qualifier: Optional[str]
    availability_status: (
        str  # "immediate", "within_month", "within_2_months", "uncertain", "not_specified"
    )
    salary_confidence: str  # "high", "medium", "low", "uncertain"
    location_preferences: List[str]
    relocation_willingness: Optional[bool]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "salary_expectation": self.salary_expectation,
            "notice_period_days": self.notice_period_days,
            "notice_period_qualifier": self.notice_period_qualifier,
            "availability_status": self.availability_status,
            "salary_confidence": self.salary_confidence,
            "location_preferences": self.location_preferences,
            "relocation_willingness": self.relocation_willingness,
        }


@dataclass
class SkillConfirmation:
    """Confirmed skills from answers and behavioral analysis."""

    confirmed_skills: List[str]
    skill_gaps: List[str]
    skill_confidence_scores: Dict[str, float]  # skill -> confidence (0-1)
    experience_level: str  # "entry", "mid", "senior", "lead"
    years_experience: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confirmed_skills": self.confirmed_skills,
            "skill_gaps": self.skill_gaps,
            "skill_confidence_scores": self.skill_confidence_scores,
            "experience_level": self.experience_level,
            "years_experience": self.years_experience,
        }


@dataclass
class ScreeningReport:
    """
    Comprehensive screening report for a single candidate.

    Combines scoring results, behavioral insights, and recruiter-facing
    summaries into a single structured object.
    """

    # Metadata.
    candidate_id: str
    job_id: str
    session_id: str
    role_id: str
    generated_at: str
    request_id: str

    # Core data.
    session_score: Dict[str, Any]
    behavioral_report: Dict[str, Any]

    # Extracted insights.
    key_answers: List[KeyAnswer]
    strength_profile: StrengthProfile
    risk_profile: RiskProfile
    compensation_insights: CompensationInsights
    skill_confirmation: SkillConfirmation

    # Red flags and warnings.
    overall_red_flags: List[str]
    overall_warnings: List[str]

    # Candidate status.
    recommendation: str  # "proceed", "hold", "reject", "insufficient_data"
    confidence_score: float  # 0-1, combined scoring confidence
    overall_score: float  # 0-1, normalized session score

    # Narratives.
    recruiter_narrative: str
    executive_summary: str

    # Formatting metadata.
    report_type: str = ReportType.DETAILED_REVIEW
    format: str = ReportFormat.JSON

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "schema_version": SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "generated_at": self.generated_at,
            "request_id": self.request_id,
            "candidate_id": self.candidate_id,
            "job_id": self.job_id,
            "session_id": self.session_id,
            "role_id": self.role_id,
            "session_score": self.session_score,
            "behavioral_report": self.behavioral_report,
            "key_answers": [k.to_dict() for k in self.key_answers],
            "strength_profile": self.strength_profile.to_dict(),
            "risk_profile": self.risk_profile.to_dict(),
            "compensation_insights": self.compensation_insights.to_dict(),
            "skill_confirmation": self.skill_confirmation.to_dict(),
            "overall_red_flags": self.overall_red_flags,
            "overall_warnings": self.overall_warnings,
            "recommendation": self.recommendation,
            "confidence_score": self.confidence_score,
            "overall_score": self.overall_score,
            "recruiter_narrative": self.recruiter_narrative,
            "executive_summary": self.executive_summary,
            "report_type": self.report_type,
            "format": self.format,
        }

    def to_recruiter_summary(self) -> str:
        """Generate a one-paragraph recruiter-friendly summary."""
        parts = [
            f"Candidate {self.candidate_id} scored {self.overall_score:.1%} "
            f"overall with {self.recommendation.upper()} recommendation."
        ]

        if self.overall_red_flags:
            parts.append(f"Key red flags: {', '.join(self.overall_red_flags[:3])}.")

        if self.overall_warnings:
            parts.append(f"Key warnings: {', '.join(self.overall_warnings[:2])}.")

        if self.compensation_insights.salary_expectation:
            parts.append(
                f"Salary expectation: ${self.compensation_insights.salary_expectation:.0f}K "
                f"({self.compensation_insights.salary_confidence} confidence)."
            )

        if self.compensation_insights.availability_status != "not_specified":
            parts.append(f"Availability: {self.compensation_insights.availability_status}.")

        if self.skill_confirmation.years_experience:
            parts.append(
                f"Experience: {self.skill_confirmation.years_experience:.1f} years "
                f"({self.skill_confirmation.experience_level} level)."
            )

        return " ".join(parts)
