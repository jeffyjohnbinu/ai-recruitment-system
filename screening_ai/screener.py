"""
screening_ai/screener.py
--------------------------
Uses an LLM to produce a short structured screening summary for a
candidate: strengths, gaps, and a recommendation. The OpenAI call is
isolated in `_call_llm` so it can be mocked easily in tests.
"""

from __future__ import annotations

from dataclasses import dataclass

from config.settings import settings
from utils.logger import get_logger

logger = get_logger("screening_ai.screener")


@dataclass
class ScreeningSummary:
    strengths: str
    gaps: str
    recommendation: str  # "advance" | "hold" | "reject"


_SYSTEM_PROMPT = (
    "You are a neutral, bias-aware recruitment screening assistant. "
    "Given a candidate resume and a job description, identify concrete "
    "strengths and gaps relative to the role, and recommend one of: "
    "advance, hold, reject. Base the assessment only on job-relevant "
    "skills and experience — do not consider name, age, gender, or "
    "any other protected characteristic."
)


def _call_llm(resume_text: str, job_description: str) -> str:
    """Isolated network call so tests can monkeypatch/mock this function."""
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.ai_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"JOB DESCRIPTION:\n{job_description}\n\nRESUME:\n{resume_text}",
            },
        ],
    )
    return response.choices[0].message.content or ""


def screen_candidate(resume_text: str, job_description: str) -> ScreeningSummary:
    """Produce a structured screening summary for one candidate."""
    logger.info("Running AI screening for candidate resume (%d chars)", len(resume_text))
    raw_output = _call_llm(resume_text, job_description)

    # NOTE: scaffold parsing — replace with structured-output / JSON mode
    # once the prompt above is upgraded to request JSON explicitly.
    summary = ScreeningSummary(
        strengths=raw_output,
        gaps="",
        recommendation="hold",
    )
    logger.info("Screening complete: recommendation=%s", summary.recommendation)
    return summary
