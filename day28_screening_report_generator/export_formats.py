"""
export_formats.py
-----------------
Multiple export formats for screening reports.

Supports JSON (structured), HTML (browser-ready), printable text,
and email-friendly formats for recruiters.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .report_format import ReportFormat, ScreeningReport

# ---- Base Exporter ---- #


class ReportExporter(ABC):
    """Abstract base class for report exporters."""

    @abstractmethod
    def export(self, report: ScreeningReport, output_path: Optional[Path] = None) -> str:
        """Export the report to the target format."""
        pass

    @abstractmethod
    def format_name(self) -> str:
        """Return the format name."""
        pass


# ---- JSON Export ---- #


class JSONExporter(ReportExporter):
    """Exports reports as JSON."""

    def export(self, report: ScreeningReport, output_path: Optional[Path] = None) -> str:
        json_str = json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
        if output_path:
            output_path.write_text(json_str, encoding="utf-8")
        return json_str

    def format_name(self) -> str:
        return "json"


# ---- HTML Export ---- #


class HTMLExporter(ReportExporter):
    """Exports reports as styled HTML for browser viewing."""

    TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Screening Report - {candidate_id}</title>
    <style>
        :root {{
            --primary: #3b82f6;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
            --gray: #6b7280;
            --bg: #f9fafb;
            --card-bg: #ffffff;
            --border: #e5e7eb;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            margin: 0;
            padding: 20px;
            color: #111827;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
        }}
        .header {{
            background: var(--card-bg);
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0 0 8px 0;
            color: var(--gray);
        }}
        .header .meta {{
            color: var(--gray);
            font-size: 14px;
        }}
        .score-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 14px;
        }}
        .score-proceed {{ background: #dcfce7; color: #166534; }}
        .score-hold {{ background: #fef3c7; color: #92400e; }}
        .score-reject {{ background: #fee2e2; color: #991b1b; }}
        .card {{
            background: var(--card-bg);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            margin: 0 0 16px 0;
            font-size: 18px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 8px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
        }}
        .stat {{
            text-align: center;
            padding: 12px;
            background: var(--bg);
            border-radius: 6px;
        }}
        .stat .value {{
            font-size: 24px;
            font-weight: 600;
            color: var(--primary);
        }}
        .stat .label {{
            font-size: 12px;
            color: var(--gray);
            text-transform: uppercase;
        }}
        .flag {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            margin-right: 4px;
            margin-bottom: 4px;
        }}
        .flag-red {{ background: #fee2e2; color: #991b1b; }}
        .flag-yellow {{ background: #fef3c7; color: #92400e; }}
        .flag-green {{ background: #dcfce7; color: #166534; }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            text-align: left;
            padding: 12px 8px;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            font-weight: 600;
            color: var(--gray);
            font-size: 13px;
            text-transform: uppercase;
        }}
        .narrative {{
            line-height: 1.6;
        }}
        .section {{
            margin-top: 24px;
        }}
        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
        }}
        .badge-soft {{ background: #f3f4f6; color: var(--gray); }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>AI Screening Report</h1>
            <div class="meta">
                Candidate: {candidate_id} | Job: {job_id} | Generated: {generated}
            </div>
        </div>

        <div class="section">
            <div class="grid">
                <div class="stat">
                    <div class="value">{overall_score:.0%}</div>
                    <div class="label">Overall Score</div>
                </div>
                <div class="stat">
                    <div class="value">{confidence:.0%}</div>
                    <div class="label">Confidence</div>
                </div>
                <div class="stat">
                    <div class="value">{communication:.0%}</div>
                    <div class="label">Communication</div>
                </div>
                <div class="stat">
                    <div class="value">{recommendation}</div>
                    <div class="label">Recommendation</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Executive Summary</h2>
            <p class="narrative">{executive_summary}</p>
        </div>

        <div class="card">
            <h2>Compensation Insights</h2>
            <table>
                <tr><th>Salary Expectation</th><td>{salary}</td></tr>
                <tr><th>Notice Period</th><td>{notice}</td></tr>
                <tr><th>Availability</th><td>{availability}</td></tr>
                <tr><th>Location</th><td>{location}</td></tr>
            </table>
        </div>

        <div class="card">
            <h2>Skill Confirmation</h2>
            <p><strong>Experience:</strong> {experience} years ({level})</p>
            <div>
                {skills}
            </div>
            <div class="section" style="margin-top: 12px;">
                <strong>Confirmed Skills:</strong><br>
                {confirmed_skills}
            </div>
            {gaps_section}
        </div>

        <div class="card">
            <h2>Strengths</h2>
            {strengths}
        </div>

        <div class="card">
            <h2>Risks & Red Flags</h2>
            {flags}
        </div>

        <div class="card">
            <h2>Key Answers</h2>
            {answers_table}
        </div>

        <div class="card">
            <h2>Recruiter Narrative</h2>
            <p class="narrative">{recruiter_narrative}</p>
        </div>
    </div>
</body>
</html>"""

    def export(self, report: ScreeningReport, output_path: Optional[Path] = None) -> str:
        html = self._build_html(report)
        if output_path:
            output_path.write_text(html, encoding="utf-8")
        return html

    def format_name(self) -> str:
        return "html"

    def _build_html(self, report: ScreeningReport) -> str:
        d = report.to_dict()

        # Recommendation badge class.
        rec = d.get("recommendation", "insufficient_data")
        rec_class = (
            "score-proceed"
            if rec == "proceed"
            else ("score-hold" if rec == "hold" else "score-reject")
        )

        # Salary.
        comp = d.get("compensation_insights", {})
        salary = (
            f"${comp.get('salary_expectation', 0):.0f}K"
            if comp.get("salary_expectation")
            else "Not specified"
        )

        # Notice period.
        notice = (
            f"{comp.get('notice_period_days')} days"
            if comp.get("notice_period_days")
            else "Not specified"
        )

        # Availability.
        avail = comp.get("availability_status", "Not specified").replace("_", " ").title()

        # Location.
        locs = comp.get("location_preferences", [])
        location = ", ".join(locs) if locs else "Not specified"

        # Skills section.
        skills = d.get("skill_confirmation", {})
        exp_years = skills.get("years_experience")
        exp_level = skills.get("experience_level", "unknown")
        experience = f"{exp_years:.1f}" if exp_years else "Not specified"

        # Skills tags.
        skill_tags = ""
        for skill in skills.get("confirmed_skills", []):
            skill_tags += f'<span class="badge badge-soft">{skill}</span> '

        # Gaps section.
        gaps = skills.get("skill_gaps", [])
        if gaps:
            gaps_section = (
                f"<div class='section'><strong>Potential Gaps:</strong> {', '.join(gaps)}</div>"
            )
        else:
            gaps_section = ""

        # Strengths.
        strength = d.get("strength_profile", {})
        strengths_items = []
        if strength.get("key_strengths"):
            for s in strength["key_strengths"]:
                strengths_items.append(f"<li>{s}</li>")
        if not strengths_items:
            strengths_items.append("<li>No notable strengths identified</li>")
        strengths = f"<ul>{''.join(strengths_items)}</ul>"

        # Flags.
        risk = d.get("risk_profile", {})
        flags_html = ""
        if risk.get("critical_risks"):
            for f in risk["critical_risks"]:
                flags_html += f'<span class="flag flag-red">{f}</span> '
        if not flags_html:
            flags_html = '<span class="flag flag-green">No critical red flags</span>'

        # Answers table.
        answers = d.get("key_answers", [])
        if answers:
            answers_rows = ""
            for a in answers[:10]:
                score_class = ""
                if a.get("score", 0) < 0.5:
                    score_class = "flag-red"
                elif a.get("score", 0) < 0.7:
                    score_class = "flag-yellow"
                answers_rows += f"""<tr>
                    <td>{a.get('question_id', '')}</td>
                    <td>{a.get('category', '')}</td>
                    <td>{a.get('score', 0):.0%}</td>
                    <td>{a.get('was_answered', True) and 'Answered' or 'Not answered'}</td>
                </tr>"""
            answers_table = f"""<table>
                <thead>
                    <tr><th>QID</th><th>Category</th><th>Score</th><th>Status</th></tr>
                </thead>
                <tbody>{answers_rows}</tbody>
            </table>"""
        else:
            answers_table = "<p>No key answers extracted</p>"

        return self.TEMPLATE.format(
            candidate_id=d.get("candidate_id", "Unknown"),
            job_id=d.get("job_id", "Unknown"),
            generated=d.get("generated_at", "")[:19],
            overall_score=d.get("overall_score", 0),
            confidence=d.get("confidence_score", 0),
            communication=strength.get("communication_strength_score", 0),
            recommendation=rec.upper(),
            executive_summary=d.get("executive_summary", "No summary available"),
            salary=salary,
            notice=notice,
            availability=avail,
            location=location,
            experience=experience,
            level=exp_level.title(),
            skill_tags=skill_tags,
            gaps_section=gaps_section,
            strengths=strengths,
            flags=flags_html,
            answers_table=answers_table,
            recruiter_narrative=d.get("recruiter_narrative", "No narrative available"),
        )


# ---- Printable Text Export ---- #


class PrintableExporter(ReportExporter):
    """Exports reports as plain text suitable for printing or terminal output."""

    SEPARATOR = "=" * 70
    SECTION = "-" * 50

    def export(self, report: ScreeningReport, output_path: Optional[Path] = None) -> str:
        lines = self._build_text(report)
        text = "\n".join(lines)
        if output_path:
            output_path.write_text(text, encoding="utf-8")
        return text

    def format_name(self) -> str:
        return "printable"

    def _build_text(self, report: ScreeningReport) -> List[str]:
        d = report.to_dict()
        lines: List[str] = []

        # Header.
        lines.append(self.SEPARATOR)
        lines.append("AI SCREENING REPORT".center(70))
        lines.append(self.SEPARATOR)
        lines.append("")
        lines.append(f"Candidate ID:     {d.get('candidate_id', 'Unknown')}")
        lines.append(f"Job ID:           {d.get('job_id', 'Unknown')}")
        lines.append(f"Session ID:       {d.get('session_id', 'Unknown')}")
        lines.append(f"Role:             {d.get('role_id', 'Unknown')}")
        lines.append(f"Generated:        {d.get('generated_at', 'Unknown')[:19]}")
        lines.append("")

        # Scores.
        lines.append(self.SECTION)
        lines.append("SCORE SUMMARY")
        lines.append(self.SECTION)
        lines.append(f"Overall Score:     {d.get('overall_score', 0):.1%}")
        lines.append(f"Confidence:        {d.get('confidence_score', 0):.1%}")
        lines.append(f"Recommendation:    {d.get('recommendation', 'unknown').upper()}")
        lines.append("")

        # Executive Summary.
        lines.append(self.SECTION)
        lines.append("EXECUTIVE SUMMARY")
        lines.append(self.SECTION)
        lines.append(d.get("executive_summary", "No summary available."))
        lines.append("")

        # Compensation.
        lines.append(self.SECTION)
        lines.append("COMPENSATION & AVAILABILITY")
        lines.append(self.SECTION)
        comp = d.get("compensation_insights", {})
        lines.append(
            f"Salary Expectation: ${comp.get('salary_expectation', 0):.0f}K"
            if comp.get("salary_expectation")
            else "Salary: Not specified"
        )
        lines.append(f"Notice Period:      {comp.get('notice_period_days', 'Not specified')} days")
        lines.append(f"Availability:       {comp.get('availability_status', 'Unknown')}")
        locs = comp.get("location_preferences", [])
        lines.append(f"Location:           {', '.join(locs) if locs else 'Not specified'}")
        lines.append("")

        # Skills.
        lines.append(self.SECTION)
        lines.append("SKILL CONFIRMATION")
        lines.append(self.SECTION)
        skills = d.get("skill_confirmation", {})
        years = skills.get("years_experience")
        lines.append(
            f"Experience:         {years:.1f} years ({skills.get('experience_level', 'unknown')})"
            if years
            else "Experience: Not specified"
        )
        confirmed = skills.get("confirmed_skills", [])
        lines.append(
            f"Confirmed Skills:   {', '.join(confirmed[:8]) if confirmed else 'None identified'}"
        )
        gaps = skills.get("skill_gaps", [])
        if gaps:
            lines.append(f"Potential Gaps:   {', '.join(gaps[:5])}")
        lines.append("")

        # Strengths & Risks.
        lines.append(self.SECTION)
        lines.append("STRENGTH PROFILE")
        lines.append(self.SECTION)
        strength = d.get("strength_profile", {})
        lines.append(f"Communication Score: {strength.get('communication_strength_score', 0):.1%}")
        lines.append(f"Clarity:             {strength.get('clarity_score', 0):.1%}")
        lines.append(f"Confidence:          {strength.get('confidence_score', 0):.1%}")
        lines.append(f"Conviction:          {strength.get('conviction_score', 0):.1%}")
        lines.append("")
        lines.append("Key Strengths:")
        for s in strength.get("key_strengths", []):
            lines.append(f"  • {s}")
        if not strength.get("key_strengths"):
            lines.append("  • No notable strengths")
        lines.append("")

        lines.append(self.SECTION)
        lines.append("RISK PROFILE")
        lines.append(self.SECTION)
        risk = d.get("risk_profile", {})
        lines.append(f"Hesitation Level:    {risk.get('hesitation_level', 'unknown')}")
        lines.append(f"Uncertainty Level:   {risk.get('uncertainty_level', 'unknown')}")
        lines.append(f"Sentiment Risk:      {risk.get('sentiment_risk', 'unknown')}")
        lines.append(f"Contradictions:      {risk.get('contradiction_count', 0)}")
        lines.append("")
        if risk.get("critical_risks"):
            lines.append("Critical Risks:")
            for r in risk.get("critical_risks", []):
                lines.append(f"  ⚠ {r}")
        else:
            lines.append("No critical risks identified.")
        lines.append("")

        # Red Flags.
        lines.append(self.SECTION)
        lines.append("RED FLAGS & WARNINGS")
        lines.append(self.SECTION)
        if d.get("overall_red_flags"):
            for f in d.get("overall_red_flags", []):
                lines.append(f"  ⚠ {f}")
        else:
            lines.append("  ✓ No red flags")
        if d.get("overall_warnings"):
            for w in d.get("overall_warnings", []):
                lines.append(f"  ! {w}")
        lines.append("")

        # Recruiter Narrative.
        lines.append(self.SECTION)
        lines.append("RECRUITER NARRATIVE")
        lines.append(self.SECTION)
        lines.append(d.get("recruiter_narrative", "No narrative available."))
        lines.append("")

        lines.append(self.SEPARATOR)

        return lines


# ---- Email Friendly Export ---- #


class EmailExporter(ReportExporter):
    """Exports reports in an email-friendly plain text format."""

    def export(self, report: ScreeningReport, output_path: Optional[Path] = None) -> str:
        d = report.to_dict()
        rec = d.get("recommendation", "unknown").upper()
        overall = d.get("overall_score", 0)

        lines = [
            f"Subject: Screening Report - {d.get('candidate_id')} - {rec}",
            "",
            "=" * 60,
            "AI SCREENING REPORT",
            "=" * 60,
            "",
            f"Candidate: {d.get('candidate_id', 'Unknown')}",
            f"Job: {d.get('job_id', 'Unknown')} ({d.get('role_id', 'Unknown')})",
            "",
            "-" * 40,
            "SUMMARY",
            "-" * 40,
            f"Overall Score:    {overall:.1%}",
            f"Recommendation:   {rec}",
            "",
        ]

        # Key insights.
        comp = d.get("compensation_insights", {})
        if comp.get("salary_expectation"):
            lines.append(
                f"Salary Expect:    ${comp.get('salary_expectation'):.0f}K ({comp.get('salary_confidence', 'uncertain')})"
            )
        if comp.get("availability_status"):
            lines.append(f"Availability:     {comp.get('availability_status')}")

        skills = d.get("skill_confirmation", {})
        if skills.get("years_experience"):
            lines.append(f"Experience:       {skills.get('years_experience'):.1f} years")

        lines.extend(
            ["", "-" * 40, "NARRATIVE", "-" * 40, d.get("recruiter_narrative", "No summary"), ""]
        )

        # Alerts.
        red_flags = d.get("overall_red_flags", [])
        warnings = d.get("overall_warnings", [])
        if red_flags:
            lines.append("RED FLAGS:")
            for f in red_flags[:5]:
                lines.append(f"  - {f}")
            lines.append("")

        if warnings:
            lines.append("WARNINGS:")
            for w in warnings[:5]:
                lines.append(f"  ! {w}")
            lines.append("")

        lines.append("=" * 60)

        text = "\n".join(lines)
        if output_path:
            output_path.write_text(text, encoding="utf-8")
        return text

    def format_name(self) -> str:
        return "email"


# ---- Factory ---- #


def get_exporter(format_name: str = "json") -> ReportExporter:
    """Factory function to get an exporter by format name."""
    exporters = {
        "json": JSONExporter,
        "html": HTMLExporter,
        "printable": PrintableExporter,
        "email": EmailExporter,
    }
    if format_name.lower() not in exporters:
        raise ValueError(f"Unknown format: {format_name}. Available: {list(exporters.keys())}")
    return exporters[format_name.lower()]()
