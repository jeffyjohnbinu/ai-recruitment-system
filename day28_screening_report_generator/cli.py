"""
cli.py
------
Day 28 deliverable — Zecpath AI Job Portal

Command-line interface for the AI Screening Report Generator.

Usage:
    # Generate report from Day 26 score JSON:
    python -m day28_screening_report_generator.cli \
        --score outputs/structured/screening_scores/cand_001__job_01__sess_001.score.json \
        --behavioral outputs/structured/behavioral_reports/cand_001__job_01__sess_001.behavioral.json \
        --output-dir outputs/structured/screening_reports \
        --format json html printable email

    # Quick demo with built-in sample data:
    python -m day28_screening_report_generator.cli --demo

    # From JSON files:
    python -m day28_screening_report_generator.cli \
        --score score.json \
        --behavioral behavioral.json \
        --candidate cand_001 \
        --job job_01 \
        --session sess_001 \
        --format json

Output formats (default: json):
    json       - Structured JSON report
    html       - Browser-ready HTML report
    printable  - Plain text printable report
    email      - Email-friendly format
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger("day28_screening_report_generator.cli")

# ---- Demo data ---- #


def _demo_session_score() -> Dict[str, Any]:
    """Built-in demo session score (Day 26 format)."""
    return {
        "candidate_id": "demo_candidate",
        "job_id": "demo_job",
        "session_id": "demo_session",
        "role_id": "software_engineer",
        "total_questions": 10,
        "answered_questions": 9,
        "mandatory_unanswered": 1,
        "normalized_score": 0.72,
        "overall_dimension_scores": {
            "clarity": 0.78,
            "relevance": 0.71,
            "completeness": 0.68,
            "consistency": 0.80,
        },
        "hard_filters_failed": [],
        "recommendation": "proceed",
        "consistency": {"score": 0.85, "is_consistent": True, "flagged_questions": []},
        "breakdown": [
            {
                "question_id": "Q-SE-001",
                "category": "introduction",
                "scoring_weight": 1,
                "is_mandatory": True,
                "was_answered": True,
                "overall_score": 0.85,
                "dimension_scores": {"clarity": 0.90, "relevance": 0.80, "completeness": 0.85},
                "explanation": "Self introduction: clear and relevant.",
            },
            {
                "question_id": "Q-SE-005",
                "category": "experience",
                "scoring_weight": 4,
                "is_mandatory": True,
                "was_answered": True,
                "overall_score": 0.60,
                "dimension_scores": {"clarity": 0.70, "relevance": 0.60, "completeness": 0.50},
                "explanation": "Experience answer: vague qualifiers noted.",
            },
            {
                "question_id": "Q-SE-017",
                "category": "salary",
                "scoring_weight": 4,
                "is_mandatory": True,
                "was_answered": True,
                "overall_score": 0.70,
                "dimension_scores": {"clarity": 0.75, "relevance": 0.65, "completeness": 0.70},
                "explanation": "Salary expectation: around 18-20 LPA.",
            },
            {
                "question_id": "Q-SE-019",
                "category": "notice_period",
                "scoring_weight": 4,
                "is_mandatory": True,
                "was_answered": True,
                "overall_score": 0.80,
                "dimension_scores": {"clarity": 0.80, "relevance": 0.80, "completeness": 0.80},
                "explanation": "Notice period: 30 days.",
            },
        ],
        "narrative": "Candidate answered 9/10 questions with a proceed recommendation.",
        "warnings": ["1 mandatory question unanswered."],
    }


def _demo_behavioral_report() -> Dict[str, Any]:
    """Built-in demo behavioral report (Day 27 format)."""
    return {
        "candidate_id": "demo_candidate",
        "job_id": "demo_job",
        "session_id": "demo_session",
        "role_id": "software_engineer",
        "session": {
            "total_answers": 9,
            "flagged_answers": 2,
            "clean_answers": 7,
            "hesitation": {
                "session_hesitation_rate": 1.5,
                "session_avg_hesitation_severity": 0.35,
                "high_hesitation_answers": [],
            },
            "pace": {
                "avg_response_length": 28.5,
                "avg_pace_wps": 2.1,
                "avg_filler_ratio": 0.053,
                "pace_distribution": {"normal": 7, "slow": 1, "fast": 1},
                "length_distribution": {"normal": 7, "short": 1, "long": 1},
            },
            "sentiment": {
                "session_sentiment_polarity": 0.42,
                "session_sentiment_label": "positive",
                "sentiment_distribution": {"positive": 7, "neutral": 2, "negative": 0},
                "avg_positive_score": 0.65,
                "avg_negative_score": 0.10,
                "avg_neutral_score": 0.25,
                "negative_sentiment_answers": [],
                "mixed_sentiment_answers": [],
            },
            "uncertainty": {
                "session_uncertainty_rate": 1.2,
                "high_uncertainty_answers": [],
            },
            "strength": {
                "session_avg_strength": 0.72,
                "session_avg_confidence": 0.75,
                "strength_distribution": {
                    "exceptional": 1,
                    "strong": 3,
                    "competent": 4,
                    "developing": 1,
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
                "raw_answer": "Yes, this is a good time to talk.",
                "hesitation": {
                    "total_patterns": 1,
                    "filler_count": 0,
                    "pause_count": 0,
                    "repetition_count": 0,
                    "repair_count": 0,
                    "false_start_count": 0,
                    "max_severity": 0.2,
                    "avg_severity": 0.2,
                    "severity_label": "minimal",
                    "flagged": False,
                },
                "pace": {
                    "word_count": 8,
                    "char_count": 42,
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
                    "polarity": 0.60,
                    "sentiment_label": "positive",
                    "positive_score": 0.75,
                    "negative_score": 0.05,
                    "neutral_score": 0.20,
                    "confidence": 0.90,
                    "emotion_signals": {"happy": 0.70, "neutral": 0.20},
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
                    "clarity": 0.85,
                    "confidence": 0.80,
                    "conviction": 0.75,
                    "engagement": 0.70,
                    "professionalism": 0.80,
                    "overall_strength": 0.80,
                    "strength_label": "exceptional",
                    "primary_strength": "clarity",
                    "primary_weakness": "engagement",
                },
                "overall_confidence_score": 0.88,
                "red_flags": [],
                "warnings": [],
            },
            {
                "question_id": "Q-SE-005",
                "category": "experience",
                "raw_answer": "I have around 5 years of total professional experience.",
                "hesitation": {
                    "total_patterns": 2,
                    "filler_count": 1,
                    "pause_count": 0,
                    "repetition_count": 0,
                    "repair_count": 1,
                    "false_start_count": 0,
                    "max_severity": 0.4,
                    "avg_severity": 0.30,
                    "severity_label": "moderate",
                    "flagged": False,
                },
                "pace": {
                    "word_count": 10,
                    "char_count": 52,
                    "sentence_count": 1,
                    "avg_sentence_length": 10.0,
                    "duration_seconds": 4.0,
                    "words_per_second": 2.50,
                    "pace_label": "normal",
                    "pace_score": 1.0,
                    "length_label": "normal",
                    "length_score": 1.0,
                    "filler_ratio": 0.20,
                    "flagged": False,
                },
                "sentiment": {
                    "polarity": 0.20,
                    "sentiment_label": "neutral",
                    "positive_score": 0.30,
                    "negative_score": 0.10,
                    "neutral_score": 0.60,
                    "confidence": 0.70,
                    "emotion_signals": {"neutral": 0.60},
                    "dominant_emotion": "neutral",
                    "flagged": False,
                },
                "uncertainty": {
                    "total_signals": 1,
                    "hedge_count": 1,
                    "doubt_count": 0,
                    "vague_quantifier_count": 0,
                    "max_severity": 0.30,
                    "avg_severity": 0.30,
                    "severity_label": "minimal",
                    "flagged": False,
                },
                "strength": {
                    "clarity": 0.60,
                    "confidence": 0.55,
                    "conviction": 0.50,
                    "engagement": 0.65,
                    "professionalism": 0.60,
                    "overall_strength": 0.58,
                    "strength_label": "developing",
                    "primary_strength": "engagement",
                    "primary_weakness": "conviction",
                },
                "overall_confidence_score": 0.58,
                "red_flags": [],
                "warnings": [],
            },
            {
                "question_id": "Q-SE-017",
                "category": "salary",
                "raw_answer": "I am looking for around 18 to 20 LPA.",
                "hesitation": {
                    "total_patterns": 1,
                    "filler_count": 0,
                    "pause_count": 0,
                    "repetition_count": 0,
                    "repair_count": 0,
                    "false_start_count": 0,
                    "max_severity": 0.15,
                    "avg_severity": 0.15,
                    "severity_label": "minimal",
                    "flagged": False,
                },
                "pace": {
                    "word_count": 9,
                    "char_count": 45,
                    "sentence_count": 1,
                    "avg_sentence_length": 9.0,
                    "duration_seconds": 3.5,
                    "words_per_second": 2.57,
                    "pace_label": "normal",
                    "pace_score": 1.0,
                    "length_label": "normal",
                    "length_score": 1.0,
                    "filler_ratio": 0.0,
                    "flagged": False,
                },
                "sentiment": {
                    "polarity": 0.10,
                    "sentiment_label": "neutral",
                    "positive_score": 0.20,
                    "negative_score": 0.05,
                    "neutral_score": 0.75,
                    "confidence": 0.80,
                    "emotion_signals": {"neutral": 0.75},
                    "dominant_emotion": "neutral",
                    "flagged": False,
                },
                "uncertainty": {
                    "total_signals": 1,
                    "hedge_count": 1,
                    "doubt_count": 0,
                    "vague_quantifier_count": 0,
                    "max_severity": 0.30,
                    "avg_severity": 0.30,
                    "severity_label": "minimal",
                    "flagged": False,
                },
                "strength": {
                    "clarity": 0.70,
                    "confidence": 0.65,
                    "conviction": 0.60,
                    "engagement": 0.70,
                    "professionalism": 0.70,
                    "overall_strength": 0.67,
                    "strength_label": "competent",
                    "primary_strength": "engagement",
                    "primary_weakness": "conviction",
                },
                "overall_confidence_score": 0.67,
                "red_flags": [],
                "warnings": [],
            },
            {
                "question_id": "Q-SE-019",
                "category": "notice_period",
                "raw_answer": "My current notice period is 30 days.",
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
                    "duration_seconds": 2.5,
                    "words_per_second": 3.20,
                    "pace_label": "normal",
                    "pace_score": 1.0,
                    "length_label": "normal",
                    "length_score": 1.0,
                    "filler_ratio": 0.0,
                    "flagged": False,
                },
                "sentiment": {
                    "polarity": 0.30,
                    "sentiment_label": "neutral",
                    "positive_score": 0.40,
                    "negative_score": 0.05,
                    "neutral_score": 0.55,
                    "confidence": 0.85,
                    "emotion_signals": {"neutral": 0.55},
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
                    "clarity": 0.80,
                    "confidence": 0.80,
                    "conviction": 0.80,
                    "engagement": 0.75,
                    "professionalism": 0.80,
                    "overall_strength": 0.80,
                    "strength_label": "exceptional",
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
        "warning_count": 1,
        "overall_confidence_score": 0.75,
        "overall_strength_score": 0.72,
        "overall_sentiment_polarity": 0.30,
        "narrative": "Candidate answered 4 questions with strong communication.",
        "warnings": ["Minor hesitation detected in 1 answer."],
    }


# ---- CLI logic ---- #


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Day 28 — AI Screening Report Generator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m day28_screening_report_generator.cli --demo
    python -m day28_screening_report_generator.cli --score score.json --behavioral behavioral.json
        --format json html --output-dir outputs/reports
        """,
    )
    p.add_argument("--score", type=Path, help="Path to Day 26 score JSON file.")
    p.add_argument("--behavioral", type=Path, help="Path to Day 27 behavioral JSON file.")
    p.add_argument("--candidate", default="demo_candidate", help="Candidate ID.")
    p.add_argument("--job", default="demo_job", help="Job ID.")
    p.add_argument("--session", default="demo_session", help="Session ID.")
    p.add_argument("--role", default="software_engineer", help="Role ID.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/structured/screening_reports"),
        help="Output directory for reports.",
    )
    p.add_argument(
        "--format",
        nargs="+",
        default=["json"],
        choices=["json", "html", "printable", "email"],
        help="Export formats (default: json).",
    )
    p.add_argument(
        "--report-type",
        default="detailed_review",
        choices=["shortlisting", "detailed_review", "comparative", "executive_summary"],
        help="Report type.",
    )
    p.add_argument("--demo", action="store_true", help="Run with built-in demo data.")
    p.add_argument("--verbose", action="store_true", help="Print full report to stdout.")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Load data.
    if args.demo:
        session_score = _demo_session_score()
        behavioral_report = _demo_behavioral_report()
        logger.info("Running demo with built-in sample data.")
    else:
        if not args.score or not args.behavioral:
            raise SystemExit("Error: specify --score and --behavioral, or use --demo.")
        session_score = json.loads(args.score.read_text(encoding="utf-8"))
        behavioral_report = json.loads(args.behavioral.read_text(encoding="utf-8"))

    # Build report.
    from .report_builder import ScreeningReportBuilder

    builder = ScreeningReportBuilder()
    report = builder.build_report(
        session_score=session_score,
        behavioral_report=behavioral_report,
        request_id=str(uuid.uuid4()),
        report_type=args.report_type,
    )

    # Export.
    from .export_formats import get_exporter

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_paths: Dict[str, str] = {}
    for fmt in args.format:
        exporter = get_exporter(fmt)
        ext_map = {
            "json": "report.json",
            "html": "report.html",
            "printable": "report.txt",
            "email": "report.email.txt",
        }
        filename = f"{report.candidate_id}__{report.job_id}__{report.session_id}.{ext_map[fmt]}"
        path = output_dir / filename
        exporter.export(report, output_path=path)
        exported_paths[fmt] = str(path)
        logger.info("Exported %s report to %s", fmt, path)

    # Print summary.
    if args.verbose or not args.format:
        print(f"\n{'=' * 60}")
        print(f"  SCREENING REPORT — {report.candidate_id} / {report.session_id}")
        print(f"{'=' * 60}")
        print(f"  Recommendation: {report.recommendation.upper()}")
        print(f"  Overall Score:  {report.overall_score:.1%}")
        print(f"  Confidence:     {report.confidence_score:.1%}")
        print(f"  Communication:  {report.strength_profile.communication_strength_score:.1%}")
        print(f"  Risks:          {len(report.risk_profile.critical_risks)}")
        print(f"  Red Flags:      {len(report.overall_red_flags)}")
        print(f"  Warnings:       {len(report.overall_warnings)}")
        if report.compensation_insights.salary_expectation:
            print(f"  Salary:         ${report.compensation_insights.salary_expectation:.0f}K")
        if report.compensation_insights.availability_status != "not_specified":
            print(f"  Availability:   {report.compensation_insights.availability_status}")
        print(
            f"  Skills:         {', '.join(report.skill_confirmation.confirmed_skills[:5]) or 'None'}"
        )
        print(f"  Exported to:    {', '.join(exported_paths.values())}")
        print(f"{'=' * 60}\n")
    else:
        print(report.to_recruiter_summary())

    # Exit code.
    if report.recommendation == "reject":
        sys.exit(1)
    elif report.recommendation == "insufficient_data":
        sys.exit(2)


if __name__ == "__main__":
    import uuid

    main()
