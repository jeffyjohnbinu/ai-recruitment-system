"""
sample_reports.py
-----------------
Built-in sample screening reports for demonstration and testing.
"""

from __future__ import annotations

from typing import Any, Dict

SAMPLE_REPORTS: Dict[str, Dict[str, Any]] = {
    "java_backend_001": {
        "candidate_id": "java_dev_001",
        "job_id": "java_backend_001",
        "session_id": "sess_java_001",
        "role_id": "java_backend_developer",
        "overall_score": 0.85,
        "recommendation": "proceed",
        "confidence_score": 0.88,
        "key_insights": {
            "years_experience": 4.5,
            "salary_expectation": 16.0,
            "notice_period": 45,
            "available_in": "within_month",
            "confirmed_skills": ["Java", "Spring Boot", "Microservices", "PostgreSQL", "Docker"],
            "communication_strength": "strong",
        },
        "summary": "Strong Java backend developer with 4.5 years of experience. "
        "Excellent communication and problem-solving skills. ",
    },
    "frontend_001": {
        "candidate_id": "fe_dev_001",
        "job_id": "react_frontend_001",
        "session_id": "sess_fe_001",
        "role_id": "react_frontend_developer",
        "overall_score": 0.65,
        "recommendation": "hold",
        "confidence_score": 0.72,
        "key_insights": {
            "years_experience": 2.5,
            "salary_expectation": 10.0,
            "notice_period": 30,
            "available_in": "immediate",
            "confirmed_skills": ["React", "JavaScript", "HTML/CSS"],
            "communication_strength": "developing",
        },
        "summary": "Decent frontend developer with 2.5 years experience. "
        "Strengths in React and JavaScript. "
        "May need further assessment for advanced requirements. ",
    },
    "fullstack_001": {
        "candidate_id": "fs_dev_001",
        "job_id": "mern_fullstack_001",
        "session_id": "sess_fs_001",
        "role_id": "mern_fullstack_developer",
        "overall_score": 0.78,
        "recommendation": "proceed",
        "confidence_score": 0.85,
        "key_insights": {
            "years_experience": 3.2,
            "salary_expectation": 14.0,
            "notice_period": 30,
            "available_in": "within_month",
            "confirmed_skills": ["MongoDB", "Express", "React", "Node.js", "REST APIs"],
            "communication_strength": "strong",
        },
        "summary": "Versatile fullstack developer with MERN stack expertise. "
        "Strong across the entire stack. "
        "Good cultural fit for agile environment.",
    },
    "senior_ios_001": {
        "candidate_id": "ios_dev_senior",
        "job_id": "ios_senior_001",
        "session_id": "sess_ios_001",
        "role_id": "ios_senior_developer",
        "overall_score": 0.92,
        "recommendation": "proceed",
        "confidence_score": 0.94,
        "key_insights": {
            "years_experience": 7.0,
            "salary_expectation": 25.0,
            "notice_period": 60,
            "available_in": "within_2_months",
            "confirmed_skills": ["Swift", "iOS SDK", "Core Data", "Testing", "App Store"],
            "communication_strength": "exceptional",
        },
        "summary": "Senior iOS developer with 7 years of experience and strong leadership. "
        "Exceptional communication and technical depth. "
        "Ready to mentor junior developers.",
    },
    "data_scientist_001": {
        "candidate_id": "ds_001",
        "job_id": "ml_engineer_001",
        "session_id": "sess_ds_001",
        "role_id": "data_scientist",
        "overall_score": 0.58,
        "recommendation": "reject",
        "confidence_score": 0.62,
        "key_insights": {
            "years_experience": 1.5,
            "salary_expectation": 8.0,
            "notice_period": 30,
            "available_in": "immediate",
            "confirmed_skills": ["Python", "Pandas", "SQL"],
            "communication_strength": "developing",
        },
        "summary": "Entry-level candidate with basic ML knowledge. "
        "Lacks advanced production ML experience. "
        "Better suited for junior or internship role.",
    },
}


def get_sample_report(job_role: str = "default") -> Dict[str, Any]:
    """Get a sample report for demonstration."""
    return SAMPLE_REPORTS.get(job_role, SAMPLE_REPORTS["java_backend_001"])


def get_all_sample_report_ids() -> list[str]:
    """Get list of all sample report identifiers."""
    return list(SAMPLE_REPORTS.keys())


def format_sample_report(report: Dict[str, Any], format_type: str = "text") -> str:
    """Format a sample report as a string."""
    if format_type == "text":
        lines = [
            f"Candidate: {report['candidate_id']}",
            f"Job: {report['job_id']}",
            f"Score: {report['overall_score']:.1%}",
            f"Recommendation: {report['recommendation'].upper()}",
            f"Confidence: {report['confidence_score']:.1%}",
            "",
            f"Summary: {report['summary']}",
        ]
        return "\n".join(lines)
    elif format_type == "json":
        import json

        return json.dumps(report, indent=2)
    else:
        return str(report)
