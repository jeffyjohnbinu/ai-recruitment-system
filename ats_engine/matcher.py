"""
ats_engine/matcher.py
----------------------
Baseline keyword-overlap matcher between a candidate resume and a
job description. Serves as a simple, dependency-light starting point;
swap in `sentence-transformers` embeddings for semantic matching later
without changing the public `match_resume_to_job` interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from config.settings import settings
from utils.logger import get_logger
from utils.validators import clamp_score

logger = get_logger("ats_engine.matcher")

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+#.]{1,}")
_STOPWORDS = {
    "the", "and", "for", "with", "you", "your", "are", "our", "will",
    "have", "this", "that", "from", "role", "team", "work", "job",
}


@dataclass
class MatchResult:
    score: float
    matched_keywords: list[str]
    is_shortlisted: bool


def _tokenize(text: str) -> set[str]:
    words = {w.lower() for w in _WORD_RE.findall(text)}
    return words - _STOPWORDS


def match_resume_to_job(resume_text: str, job_description: str) -> MatchResult:
    """
    Compute a simple overlap-based match score between a resume and a
    job description. Returns a MatchResult with score in [0, 1].
    """
    resume_tokens = _tokenize(resume_text)
    job_tokens = _tokenize(job_description)

    if not job_tokens:
        logger.warning("Job description produced no keywords to match against.")
        return MatchResult(score=0.0, matched_keywords=[], is_shortlisted=False)

    matched = sorted(resume_tokens & job_tokens)
    raw_score = len(matched) / len(job_tokens)
    score = clamp_score(raw_score)

    is_shortlisted = score >= settings.ats_min_match_score
    logger.info(
        "Match score=%.2f (%d/%d keywords) shortlisted=%s",
        score, len(matched), len(job_tokens), is_shortlisted,
    )

    return MatchResult(score=score, matched_keywords=matched, is_shortlisted=is_shortlisted)
