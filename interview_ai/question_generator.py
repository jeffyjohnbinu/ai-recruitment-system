"""
interview_ai/question_generator.py
------------------------------------
Generates role-relevant interview questions from a job description
and (optionally) a candidate's resume, so interviewers get a
consistent, job-focused starting point.
"""

from __future__ import annotations

from utils.logger import get_logger

logger = get_logger("interview_ai.question_generator")

_DEFAULT_QUESTION_COUNT = 5


def generate_questions(
    job_description: str,
    resume_text: str | None = None,
    count: int = _DEFAULT_QUESTION_COUNT,
) -> list[str]:
    """
    Generate `count` interview questions tailored to the job description
    and, if provided, the candidate's resume.

    This scaffold returns deterministic placeholder questions so the
    module is testable without network access. Wire in an LLM call
    (see screening_ai/screener.py for the pattern) to make it dynamic.
    """
    logger.info("Generating %d interview questions", count)

    base_questions = [
        "Walk me through a project most relevant to this role.",
        "What was the most challenging technical decision you made recently, and why?",
        "How do you approach collaborating with team members who disagree with you?",
        "Describe a time you had to learn a new tool or technology quickly.",
        "What metrics or outcomes do you use to judge your own success in a role like this?",
    ]

    if resume_text:
        base_questions.append(
            "I noticed some specific experience on your resume — can you expand on it?"
        )

    questions = base_questions[:count]
    logger.info("Generated %d questions", len(questions))
    return questions
