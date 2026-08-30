"""
fixtures.py
------------
Test dataset for Day 17 ATS System Testing.

Each ATSTestCase pairs a resume with a job description and carries a
`ground_truth` block produced by MANUAL REVIEW (a human recruiter reading
both documents). The harness runs the same pair through the automated
ATS pipeline and compares the two.

Categories are deliberately balanced across the two axes called out in
the Day 17 brief:
    - role type   : tech / non-tech
    - seniority    : fresher / senior
and, within each cell, one "should match" and one "should NOT match"
case, so precision and recall are both exercised (not just accuracy on
easy positives).

Candidate "Rahul Verma" is reused as the project's standard placeholder
where a neutral filler identity is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GroundTruth:
    """Manually-reviewed (human recruiter) judgment for one resume/JD pair."""

    shortlisted: bool
    recommendation: str  # "advance" | "hold" | "reject"
    expected_matched_skills: List[str]
    reviewer_notes: str


@dataclass
class ATSTestCase:
    case_id: str
    role_type: str  # "tech" | "non_tech"
    seniority: str  # "fresher" | "senior"
    expected_outcome: str  # "match" | "mismatch" -- the scenario this case is designed to probe
    candidate_name: str
    resume_text: str
    job_description: str
    ground_truth: GroundTruth
    tags: List[str] = field(default_factory=list)


TEST_CASES: List[ATSTestCase] = [
    # ------------------------------------------------------------------ #
    # TECH — SENIOR
    # ------------------------------------------------------------------ #
    ATSTestCase(
        case_id="TECH-SR-01-MATCH",
        role_type="tech",
        seniority="senior",
        expected_outcome="match",
        candidate_name="Rahul Verma",
        resume_text=(
            "RAHUL VERMA\n"
            "rahul.verma@email.com | +91-9876543210\n\n"
            "Summary\n"
            "Senior Backend Engineer with 8 years of experience building "
            "scalable microservices in Python and Go. Led teams of 4-6 "
            "engineers, owns production reliability for payment systems.\n\n"
            "Experience\n"
            "Staff Backend Engineer -- FinPay (2019-Present)\n"
            "- Designed REST and gRPC APIs serving 5M+ daily transactions\n"
            "- Migrated monolith to Kubernetes-based microservices\n"
            "- Mentored 5 engineers; ran the on-call rotation\n"
            "Backend Engineer -- Acme Corp (2015-2019)\n"
            "- Built PostgreSQL-backed billing service, 99.99% uptime\n\n"
            "Education\n"
            "M.Tech in Computer Science -- IIT Bombay, 2015\n\n"
            "Skills\n"
            "Python, Go, Kubernetes, Docker, PostgreSQL, AWS, gRPC, Kafka, CI/CD"
        ),
        job_description=(
            "Senior Backend Engineer (5+ years)\n"
            "We are hiring a Senior Backend Engineer to own our payments "
            "microservices platform. Requirements: 5+ years backend "
            "experience, strong Python or Go, Kubernetes and Docker in "
            "production, experience with PostgreSQL and distributed "
            "systems, prior mentoring or tech-lead experience preferred."
        ),
        ground_truth=GroundTruth(
            shortlisted=True,
            recommendation="advance",
            expected_matched_skills=["python", "kubernetes", "docker", "postgresql"],
            reviewer_notes=(
                "Strong seniority and stack match; exceeds experience bar and has "
                "direct payments-domain and mentoring experience."
            ),
        ),
        tags=["backend", "seniority-match", "stack-match"],
    ),
    ATSTestCase(
        case_id="TECH-SR-02-MISMATCH",
        role_type="tech",
        seniority="senior",
        expected_outcome="mismatch",
        candidate_name="Priya Nair",
        resume_text=(
            "PRIYA NAIR\n"
            "priya.nair@email.com | +91-9123456780\n\n"
            "Summary\n"
            "Senior mobile app designer with 9 years crafting iOS "
            "interfaces and design systems for consumer apps.\n\n"
            "Experience\n"
            "Lead Product Designer -- Globex (2018-Present)\n"
            "- Owned design system used across 6 iOS apps\n"
            "- Ran user research and usability testing programs\n"
            "Visual Designer -- Initech (2014-2018)\n"
            "- Designed marketing sites and app store creatives\n\n"
            "Education\n"
            "B.Des in Visual Communication -- NID Ahmedabad, 2014\n\n"
            "Skills\n"
            "Figma, Sketch, iOS Human Interface Guidelines, Prototyping, "
            "User Research"
        ),
        job_description=(
            "Senior Backend Engineer (5+ years)\n"
            "We are hiring a Senior Backend Engineer to own our payments "
            "microservices platform. Requirements: 5+ years backend "
            "experience, strong Python or Go, Kubernetes and Docker in "
            "production, experience with PostgreSQL and distributed "
            "systems, prior mentoring or tech-lead experience preferred."
        ),
        ground_truth=GroundTruth(
            shortlisted=False,
            recommendation="reject",
            expected_matched_skills=[],
            reviewer_notes=(
                "Seniority band matches but domain is entirely different "
                "(product design vs backend engineering). No relevant "
                "technical stack overlap; should not be shortlisted."
            ),
        ),
        tags=["backend", "domain-mismatch"],
    ),
    # ------------------------------------------------------------------ #
    # TECH — FRESHER
    # ------------------------------------------------------------------ #
    ATSTestCase(
        case_id="TECH-FR-01-MATCH",
        role_type="tech",
        seniority="fresher",
        expected_outcome="match",
        candidate_name="Arjun Mehta",
        resume_text=(
            "ARJUN MEHTA\n"
            "arjun.mehta@email.com | +91-9988776655\n\n"
            "Summary\n"
            "Final-year Computer Science student with internship experience "
            "in full-stack web development using Python and JavaScript.\n\n"
            "Experience\n"
            "Software Engineering Intern -- Beta Inc (Summer 2025)\n"
            "- Built REST APIs with FastAPI and wrote pytest test suites\n"
            "- Contributed to a React front-end for an internal dashboard\n\n"
            "Projects\n"
            "- Built a resume parser using Python and spaCy for a class project\n"
            "- Built a personal portfolio site with React and Tailwind CSS\n\n"
            "Education\n"
            "B.Tech in Computer Science -- VIT Vellore, 2026 (expected)\n\n"
            "Skills\n"
            "Python, JavaScript, FastAPI, React, Git, SQL, REST APIs"
        ),
        job_description=(
            "Junior Software Engineer (0-2 years / New Grad)\n"
            "Looking for a new-grad or junior engineer comfortable with "
            "Python and JavaScript, familiar with REST API development. "
            "Internship experience with FastAPI, Flask, or Django and some "
            "React exposure is a strong plus. No prior full-time experience "
            "required."
        ),
        ground_truth=GroundTruth(
            shortlisted=True,
            recommendation="advance",
            expected_matched_skills=["python", "javascript", "fastapi", "react"],
            reviewer_notes=(
                "Exactly the target profile for a new-grad role: relevant "
                "internship, matching stack, appropriate experience level."
            ),
        ),
        tags=["fresher", "stack-match"],
    ),
    ATSTestCase(
        case_id="TECH-FR-02-MISMATCH",
        role_type="tech",
        seniority="fresher",
        expected_outcome="mismatch",
        candidate_name="Sneha Iyer",
        resume_text=(
            "SNEHA IYER\n"
            "sneha.iyer@email.com | +91-9871234560\n\n"
            "Summary\n"
            "Recent graduate in Business Administration with internship "
            "experience in digital marketing and social media management.\n\n"
            "Experience\n"
            "Marketing Intern -- Globex (Summer 2025)\n"
            "- Ran Instagram and LinkedIn campaigns, grew followers by 20%\n"
            "- Built monthly performance reports in Excel\n\n"
            "Education\n"
            "BBA in Marketing -- Christ University, 2026 (expected)\n\n"
            "Skills\n"
            "Social Media Marketing, Excel, Canva, Content Writing"
        ),
        job_description=(
            "Junior Software Engineer (0-2 years / New Grad)\n"
            "Looking for a new-grad or junior engineer comfortable with "
            "Python and JavaScript, familiar with REST API development. "
            "Internship experience with FastAPI, Flask, or Django and some "
            "React exposure is a strong plus. No prior full-time experience "
            "required."
        ),
        ground_truth=GroundTruth(
            shortlisted=False,
            recommendation="reject",
            expected_matched_skills=[],
            reviewer_notes=(
                "No software engineering background or coding skills on the "
                "resume; correct experience band but wrong domain entirely."
            ),
        ),
        tags=["fresher", "domain-mismatch"],
    ),
    # ------------------------------------------------------------------ #
    # NON-TECH — SENIOR
    # ------------------------------------------------------------------ #
    ATSTestCase(
        case_id="NONTECH-SR-01-MATCH",
        role_type="non_tech",
        seniority="senior",
        expected_outcome="match",
        candidate_name="Kavita Rao",
        resume_text=(
            "KAVITA RAO\n"
            "kavita.rao@email.com | +91-9765432109\n\n"
            "Summary\n"
            "HR leader with 10 years in talent acquisition and HR business "
            "partnering for 500+ person technology organizations.\n\n"
            "Experience\n"
            "Head of Talent Acquisition -- Globex (2020-Present)\n"
            "- Built out recruiting function from 2 to 12 recruiters\n"
            "- Owns hiring for engineering, sales, and G&A functions\n"
            "- Implemented structured interviewing and DEI hiring practices\n"
            "HR Manager -- Initech (2014-2020)\n"
            "- Managed full-cycle recruiting and employee relations\n\n"
            "Education\n"
            "MBA in Human Resources -- XLRI Jamshedpur, 2014\n\n"
            "Skills\n"
            "Talent Acquisition, HR Business Partnering, ATS Administration, "
            "Structured Interviewing, Employee Relations, DEI Programs"
        ),
        job_description=(
            "Director of Talent Acquisition (8+ years)\n"
            "Seeking a senior talent acquisition leader to build and scale "
            "our recruiting org. Requirements: 8+ years in talent "
            "acquisition or HR business partnering, experience scaling a "
            "recruiting team, ATS administration experience, background "
            "implementing structured/DEI-aligned hiring practices."
        ),
        ground_truth=GroundTruth(
            shortlisted=True,
            recommendation="advance",
            expected_matched_skills=[
                "talent acquisition",
                "hr business partnering",
                "ats administration",
                "structured interviewing",
            ],
            reviewer_notes=(
                "Direct functional and seniority match; has scaled a recruiting org before."
            ),
        ),
        tags=["hr", "seniority-match"],
    ),
    ATSTestCase(
        case_id="NONTECH-SR-02-MISMATCH",
        role_type="non_tech",
        seniority="senior",
        expected_outcome="mismatch",
        candidate_name="Vikram Shah",
        resume_text=(
            "VIKRAM SHAH\n"
            "vikram.shah@email.com | +91-9012345678\n\n"
            "Summary\n"
            "Senior enterprise sales leader with 11 years closing large "
            "SaaS deals across BFSI and retail accounts.\n\n"
            "Experience\n"
            "VP of Sales -- Acme Corp (2019-Present)\n"
            "- Owns $40M ARR enterprise sales pipeline\n"
            "- Built and led a team of 15 account executives\n"
            "Regional Sales Manager -- Beta Inc (2013-2019)\n"
            "- Grew territory revenue by 3x over 4 years\n\n"
            "Education\n"
            "MBA in Sales & Marketing -- IIM Ahmedabad, 2013\n\n"
            "Skills\n"
            "Enterprise Sales, Salesforce CRM, Negotiation, Account "
            "Management, Sales Forecasting"
        ),
        job_description=(
            "Director of Talent Acquisition (8+ years)\n"
            "Seeking a senior talent acquisition leader to build and scale "
            "our recruiting org. Requirements: 8+ years in talent "
            "acquisition or HR business partnering, experience scaling a "
            "recruiting team, ATS administration experience, background "
            "implementing structured/DEI-aligned hiring practices."
        ),
        ground_truth=GroundTruth(
            shortlisted=False,
            recommendation="reject",
            expected_matched_skills=[],
            reviewer_notes=(
                "Seniority band and leadership scope match well, but the "
                "functional background (sales) has no transferable overlap "
                "with talent acquisition / HR."
            ),
        ),
        tags=["hr", "domain-mismatch"],
    ),
    # ------------------------------------------------------------------ #
    # NON-TECH — FRESHER
    # ------------------------------------------------------------------ #
    ATSTestCase(
        case_id="NONTECH-FR-01-MATCH",
        role_type="non_tech",
        seniority="fresher",
        expected_outcome="match",
        candidate_name="Neha Kulkarni",
        resume_text=(
            "NEHA KULKARNI\n"
            "neha.kulkarni@email.com | +91-9345678901\n\n"
            "Summary\n"
            "Recent marketing graduate with internship experience in "
            "content marketing and campaign analytics.\n\n"
            "Experience\n"
            "Marketing Intern -- Acme Corp (Summer 2025)\n"
            "- Wrote blog and email content, tracked campaign metrics in "
            "Google Analytics\n"
            "- Assisted with social media calendar and reporting\n\n"
            "Education\n"
            "BBA in Marketing -- Symbiosis Pune, 2026 (expected)\n\n"
            "Skills\n"
            "Content Marketing, Google Analytics, Email Marketing, Excel, "
            "Social Media"
        ),
        job_description=(
            "Marketing Associate (0-2 years / New Grad)\n"
            "Looking for an entry-level marketing associate to support "
            "content and campaign execution. Familiarity with Google "
            "Analytics, email marketing tools, and social media reporting "
            "preferred. No prior full-time experience required."
        ),
        ground_truth=GroundTruth(
            shortlisted=True,
            recommendation="advance",
            expected_matched_skills=[
                "content marketing",
                "google analytics",
                "email marketing",
            ],
            reviewer_notes=(
                "Good entry-level fit; internship directly maps to the role's core tasks."
            ),
        ),
        tags=["marketing", "fresher", "stack-match"],
    ),
    ATSTestCase(
        case_id="NONTECH-FR-02-MISMATCH",
        role_type="non_tech",
        seniority="fresher",
        expected_outcome="mismatch",
        candidate_name="Aditya Kumar",
        resume_text=(
            "ADITYA KUMAR\n"
            "aditya.kumar@email.com | +91-9456789012\n\n"
            "Summary\n"
            "Recent commerce graduate seeking an entry-level role; no "
            "prior internship or work experience.\n\n"
            "Education\n"
            "B.Com -- Delhi University, 2026 (expected)\n\n"
            "Skills\n"
            "MS Office, Basic Excel, Tally"
        ),
        job_description=(
            "Senior Financial Controller (10+ years)\n"
            "Seeking a senior financial controller to lead our accounting "
            "close process. Requirements: 10+ years in accounting/finance, "
            "CA/CPA qualification required, experience managing statutory "
            "audits and a team of accountants."
        ),
        ground_truth=GroundTruth(
            shortlisted=False,
            recommendation="reject",
            expected_matched_skills=[],
            reviewer_notes=(
                "Massive experience-level gap (fresher vs. 10+ year senior "
                "leadership role) and missing the required CA/CPA "
                "qualification; correctly a hard reject."
            ),
        ),
        tags=["finance", "seniority-mismatch", "experience-gap"],
    ),
]


def get_test_cases(
    role_type: Optional[str] = None, seniority: Optional[str] = None
) -> List[ATSTestCase]:
    """Filter the fixture set by role_type and/or seniority."""
    cases = TEST_CASES
    if role_type:
        cases = [c for c in cases if c.role_type == role_type]
    if seniority:
        cases = [c for c in cases if c.seniority == seniority]
    return cases
