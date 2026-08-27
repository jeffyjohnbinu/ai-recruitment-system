"""
explainability.py
------------------
Turns a ScoringResult into recruiter-readable explanations: a one-paragraph
narrative summary plus a per-component explanation string. Kept separate
from scoring_engine.py so the scoring math and the human-facing wording can
evolve independently.
"""

from __future__ import annotations

from typing import Dict

from .scoring_engine import ScoringResult

_LABELS = {
    "skill_match": "skill match",
    "experience_relevance": "experience relevance",
    "education_alignment": "education alignment",
    "semantic_similarity": "semantic similarity",
}


def build_narrative(result: ScoringResult) -> str:
    if result.status == "insufficient_data":
        missing = ", ".join(_LABELS[n] for n in result.components_missing)
        return (
            f"Candidate {result.candidate_id} could not be scored against job "
            f"{result.job_id}: too many required inputs were missing ({missing})."
        )

    # Rank by raw_score (not weighted contribution) so "strongest"/"weakest"
    # reflects how the candidate actually performed on each signal, rather
    # than being skewed by how heavily a role profile happens to weight it.
    available = [c for c in result.components if c.available]
    available_sorted = sorted(available, key=lambda c: c.raw_score or 0.0, reverse=True)
    top_driver = available_sorted[0] if available_sorted else None
    weak_driver = available_sorted[-1] if len(available_sorted) > 1 else None

    pct = round((result.final_score or 0.0) * 100)
    sentence = (
        f"Candidate {result.candidate_id} scored {pct}% overall against job "
        f"{result.job_id} (role profile: {result.role_profile})"
    )

    if top_driver is not None:
        sentence += (
            f", driven primarily by strong {_LABELS[top_driver.name]} "
            f"({round((top_driver.raw_score or 0.0) * 100)}%)"
        )
    if weak_driver is not None and weak_driver is not top_driver:
        sentence += (
            f", while {_LABELS[weak_driver.name]} was comparatively weaker "
            f"({round((weak_driver.raw_score or 0.0) * 100)}%)"
        )
    sentence += "."

    if result.components_missing:
        missing_labels = ", ".join(_LABELS[n] for n in result.components_missing)
        sentence += f" Note: {missing_labels} data was unavailable; "
        sentence += (
            "remaining weights were renormalized across the available components."
            if result.weights_renormalized
            else "a neutral default score was used in its place."
        )

    return sentence


def build_component_explanations(result: ScoringResult) -> Dict[str, str]:
    explanations: Dict[str, str] = {}
    for c in result.components:
        label = _LABELS[c.name]
        if c.available:
            explanations[c.name] = (
                f"{label}: {round((c.raw_score or 0.0) * 100)}% raw score, weighted at "
                f"{round(c.weight_applied * 100)}% -> contributed "
                f"{round(c.weighted_contribution * 100, 1)} points to the final score."
            )
        else:
            reason = "; ".join(c.notes) or "no source data found"
            explanations[c.name] = f"{label}: data unavailable ({reason})."
    return explanations
