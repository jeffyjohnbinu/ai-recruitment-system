"""
Experience Relevance Scoring
-----------------------------
Scores how relevant each parsed work-experience entry is to a target
job (title + keyword set), and provides a general-purpose role-to-role
title similarity function usable outside the scoring pipeline (e.g.
career-progression / job-hopping analysis).

Designed to compose with:
  - `experience_parsing_engine.parser.ExperienceEntry` (this package)
  - the keyword/requirement lists produced by the Day 6 JD Parsing
    Engine (`jd_parsing_engine`) — pass its extracted required_skills /
    keywords list straight into `score_experience(..., job_keywords=...)`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from .parser import ExperienceEntry

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+#.]{1,}")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "you",
    "your",
    "are",
    "our",
    "will",
    "have",
    "this",
    "that",
    "from",
    "role",
    "team",
    "work",
    "job",
    "of",
    "in",
    "on",
    "at",
    "to",
    "a",
    "an",
    "as",
    "is",
    "was",
    "were",
}

# Seniority ladder used to score how "close" two titles are in level,
# independent of raw token overlap (e.g. "Staff Engineer" vs "Principal
# Engineer" share no tokens but are adjacent in seniority).
_SENIORITY_LADDER = {
    "intern": 0,
    "trainee": 0,
    "junior": 1,
    "associate": 1,
    "entry": 1,
    "mid": 2,
    "senior": 3,
    "sr": 3,
    "lead": 3.5,
    "staff": 4,
    "principal": 4.5,
    "director": 5.5,
    "head": 5.5,
    "vp": 6.5,
    "vice president": 6.5,
    "chief": 7.5,
    "cxo": 7.5,
}
_DEFAULT_SENIORITY = 2.0  # unspecified level assumed "mid"
_SENIORITY_SCALE = 7.5  # max value in the ladder, for normalizing distance


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)} - _STOPWORDS


def _seniority_level(title: str) -> float:
    lowered = title.lower()
    for token, level in _SENIORITY_LADDER.items():
        if token in lowered:
            return level
    return _DEFAULT_SENIORITY


def title_similarity(title_a: str, title_b: str) -> float:
    """
    General-purpose role-to-role similarity in [0, 1], combining:
      - token (Jaccard) overlap between the two titles (70%)
      - closeness on a seniority ladder (30%)

    Usable standalone for career-progression comparisons, not just
    for job-relevance scoring.
    """
    tokens_a = _tokenize(title_a)
    tokens_b = _tokenize(title_b)

    if not tokens_a or not tokens_b:
        jaccard = 0.0
    else:
        union = tokens_a | tokens_b
        jaccard = len(tokens_a & tokens_b) / len(union) if union else 0.0

    level_a = _seniority_level(title_a)
    level_b = _seniority_level(title_b)
    seniority_closeness = 1.0 - min(abs(level_a - level_b) / _SENIORITY_SCALE, 1.0)

    return round((jaccard * 0.7) + (seniority_closeness * 0.3), 4)


@dataclass
class RoleRelevanceScore:
    title: str
    company: str
    duration_months: int
    title_similarity_score: float
    keyword_overlap_score: float
    relevance_score: float
    matched_keywords: List[str] = field(default_factory=list)


@dataclass
class ExperienceRelevanceResult:
    target_title: str
    overall_relevance_score: float
    relevant_experience_years: float
    total_experience_years: float
    role_scores: List[RoleRelevanceScore]
    candidate_id: Optional[str] = None
    job_id: Optional[str] = None


_TITLE_WEIGHT = 0.5
_KEYWORD_WEIGHT = 0.5
_RELEVANT_THRESHOLD = 0.4  # role counts toward "relevant experience" at/above this score


class ExperienceRelevanceScorer:
    """Scores parsed ExperienceEntry objects against a target job."""

    def score_experience(
        self,
        entries: Iterable[ExperienceEntry],
        target_title: str,
        job_keywords: Iterable[str],
        candidate_id: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> ExperienceRelevanceResult:
        job_keyword_set = {k.lower() for k in job_keywords}
        entries = list(entries)

        role_scores: List[RoleRelevanceScore] = []
        for entry in entries:
            role_scores.append(self._score_role(entry, target_title, job_keyword_set))

        total_months = sum(e.duration_months for e in entries)
        relevant_months = sum(
            rs.duration_months for rs in role_scores if rs.relevance_score >= _RELEVANT_THRESHOLD
        )

        overall = self._weighted_overall(role_scores)

        return ExperienceRelevanceResult(
            target_title=target_title,
            overall_relevance_score=overall,
            relevant_experience_years=round(relevant_months / 12, 2),
            total_experience_years=round(total_months / 12, 2),
            role_scores=role_scores,
            candidate_id=candidate_id,
            job_id=job_id,
        )

    def _score_role(
        self, entry: ExperienceEntry, target_title: str, job_keyword_set: set[str]
    ) -> RoleRelevanceScore:
        t_sim = title_similarity(entry.title, target_title)

        role_text_tokens = _tokenize(f"{entry.title} {entry.description}")
        matched = sorted(role_text_tokens & job_keyword_set)
        keyword_score = len(matched) / len(job_keyword_set) if job_keyword_set else 0.0
        keyword_score = min(keyword_score, 1.0)

        combined = round((t_sim * _TITLE_WEIGHT) + (keyword_score * _KEYWORD_WEIGHT), 4)

        return RoleRelevanceScore(
            title=entry.title,
            company=entry.company,
            duration_months=entry.duration_months,
            title_similarity_score=t_sim,
            keyword_overlap_score=round(keyword_score, 4),
            relevance_score=combined,
            matched_keywords=matched,
        )

    @staticmethod
    def _weighted_overall(role_scores: List[RoleRelevanceScore]) -> float:
        total_weight = sum(rs.duration_months for rs in role_scores)
        if total_weight == 0:
            # No parseable durations -- fall back to a simple average
            # so a relevance score is still produced.
            if not role_scores:
                return 0.0
            return round(sum(rs.relevance_score for rs in role_scores) / len(role_scores), 4)

        weighted_sum = sum(rs.relevance_score * rs.duration_months for rs in role_scores)
        return round(weighted_sum / total_weight, 4)
