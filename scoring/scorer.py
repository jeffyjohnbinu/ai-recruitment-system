"""
scoring/scorer.py
-------------------
Combines the ATS keyword-match score with an AI screening
recommendation into a single, weighted final candidate score, used
for ranking candidates in a shortlist.
"""

from __future__ import annotations

from dataclasses import dataclass

from utils.logger import get_logger
from utils.validators import clamp_score

logger = get_logger("scoring.scorer")

_RECOMMENDATION_WEIGHTS = {
    "advance": 1.0,
    "hold": 0.5,
    "reject": 0.0,
}

# How much weight the ATS keyword score vs. the AI recommendation
# carries in the final score. Must sum to 1.0.
_ATS_WEIGHT = 0.4
_AI_WEIGHT = 0.6


@dataclass
class FinalScore:
    ats_score: float
    ai_recommendation: str
    final_score: float


def compute_final_score(ats_score: float, ai_recommendation: str) -> FinalScore:
    """
    Blend the ATS match score and AI screening recommendation into a
    single final score in [0, 1], used to rank candidates.
    """
    ai_component = _RECOMMENDATION_WEIGHTS.get(ai_recommendation, 0.5)
    combined = (ats_score * _ATS_WEIGHT) + (ai_component * _AI_WEIGHT)
    final = clamp_score(combined)

    logger.info(
        "Final score computed: ats=%.2f ai_rec=%s -> final=%.2f",
        ats_score, ai_recommendation, final,
    )
    return FinalScore(ats_score=ats_score, ai_recommendation=ai_recommendation, final_score=final)


def rank_candidates(scores: list[FinalScore]) -> list[FinalScore]:
    """Sort candidates by final_score, descending (best first)."""
    return sorted(scores, key=lambda s: s.final_score, reverse=True)
