"""
scoring_engine.py
------------------
Core ATS scoring formula: combines skill match, experience relevance,
education alignment, and semantic similarity into a single, weighted,
explainable candidate score in [0, 1].

Missing-data handling modes:
    "renormalize"  (default) -- drop missing components and renormalize the
                    remaining weights so they still sum to 1.0.
    "neutral_fill" -- fill missing components with a neutral score
                    (default 0.5) at their full configured weight.
    "strict"       -- refuse to score (status="insufficient_data") if more
                    than `max_missing_for_strict` components are missing;
                    otherwise behaves like "renormalize" for the remainder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .loaders import (
    ComponentExtractionResult,
    extract_education_alignment,
    extract_experience_relevance,
    extract_semantic_similarity,
    extract_skill_match,
)
from .weights import COMPONENT_NAMES, WeightProfileRegistry

NEUTRAL_FILL_DEFAULT = 0.5
MISSING_DATA_MODES = ("renormalize", "neutral_fill", "strict")

_EXTRACTORS = {
    "skill_match": extract_skill_match,
    "experience_relevance": extract_experience_relevance,
    "education_alignment": extract_education_alignment,
    "semantic_similarity": extract_semantic_similarity,
}


@dataclass
class ComponentBreakdown:
    name: str
    available: bool
    raw_score: Optional[float]
    weight_original: float
    weight_applied: float
    weighted_contribution: float
    notes: List[str] = field(default_factory=list)


@dataclass
class ScoringResult:
    candidate_id: str
    job_id: str
    role_profile: str
    final_score: Optional[float]
    status: str  # "scored" | "insufficient_data"
    components: List[ComponentBreakdown]
    missing_data_mode: str
    components_missing: List[str]
    weights_renormalized: bool
    warnings: List[str]


class ATSScoringEngine:
    """Transparent, explainable ATS scoring engine (Day 13 deliverable)."""

    def __init__(
        self,
        registry: Optional[WeightProfileRegistry] = None,
        missing_data_mode: str = "renormalize",
        neutral_fill_value: float = NEUTRAL_FILL_DEFAULT,
        max_missing_for_strict: int = 1,
    ):
        if missing_data_mode not in MISSING_DATA_MODES:
            raise ValueError(f"missing_data_mode must be one of {MISSING_DATA_MODES}")
        if not (0.0 <= neutral_fill_value <= 1.0):
            raise ValueError("neutral_fill_value must be in [0, 1]")

        self.registry = registry or WeightProfileRegistry()
        self.missing_data_mode = missing_data_mode
        self.neutral_fill_value = neutral_fill_value
        self.max_missing_for_strict = max_missing_for_strict

    def compute_score(
        self,
        candidate_id: str,
        job_id: str,
        skill_data: Optional[dict] = None,
        experience_data: Optional[dict] = None,
        education_data: Optional[dict] = None,
        semantic_data: Optional[dict] = None,
        role: Optional[str] = None,
        score_overrides: Optional[Dict[str, float]] = None,
    ) -> ScoringResult:
        """
        Compute an explainable ATS score for one candidate/job pair.

        `score_overrides` lets a caller supply an already-known component
        score (e.g. 0-1 float) directly, bypassing extraction from raw
        upstream JSON -- useful for pipeline integration or testing.
        """
        profile = self.registry.get(role)
        score_overrides = score_overrides or {}
        raw_inputs = {
            "skill_match": skill_data,
            "experience_relevance": experience_data,
            "education_alignment": education_data,
            "semantic_similarity": semantic_data,
        }

        extraction_map: Dict[str, ComponentExtractionResult] = {}
        for name in COMPONENT_NAMES:
            if name in score_overrides:
                extraction_map[name] = ComponentExtractionResult(
                    float(score_overrides[name]), True, source_fields_used=["explicit override"]
                )
            else:
                extraction_map[name] = _EXTRACTORS[name](raw_inputs[name])

        components_missing = [n for n in COMPONENT_NAMES if not extraction_map[n].available]
        mode = self.missing_data_mode
        warnings: List[str] = []

        if mode == "strict" and len(components_missing) > self.max_missing_for_strict:
            components = [
                ComponentBreakdown(
                    name=n,
                    available=extraction_map[n].available,
                    raw_score=extraction_map[n].score,
                    weight_original=profile.weights[n],
                    weight_applied=0.0,
                    weighted_contribution=0.0,
                    notes=extraction_map[n].notes,
                )
                for n in COMPONENT_NAMES
            ]
            return ScoringResult(
                candidate_id=candidate_id,
                job_id=job_id,
                role_profile=profile.name,
                final_score=None,
                status="insufficient_data",
                components=components,
                missing_data_mode=mode,
                components_missing=components_missing,
                weights_renormalized=False,
                warnings=[
                    f"Too many missing components ({len(components_missing)} of "
                    f"{len(COMPONENT_NAMES)}) for strict mode "
                    f"(tolerance={self.max_missing_for_strict})."
                ],
            )

        applied_weights: Dict[str, float] = dict(profile.weights)
        weights_renormalized = False

        if components_missing and mode in ("renormalize", "strict"):
            available_names = [n for n in COMPONENT_NAMES if n not in components_missing]
            available_weight_sum = sum(profile.weights[n] for n in available_names)
            if available_weight_sum > 0:
                applied_weights = {
                    n: (profile.weights[n] / available_weight_sum if n in available_names else 0.0)
                    for n in COMPONENT_NAMES
                }
                weights_renormalized = True
            else:
                applied_weights = {n: 0.0 for n in COMPONENT_NAMES}
            warnings.append(
                f"Missing components {components_missing} -- weights renormalized "
                f"among the {len(available_names)} available component(s)."
            )
        elif components_missing and mode == "neutral_fill":
            warnings.append(
                f"Missing components {components_missing} -- filled with neutral score "
                f"{self.neutral_fill_value} at full configured weight."
            )

        components: List[ComponentBreakdown] = []
        final_score = 0.0

        for name in COMPONENT_NAMES:
            result = extraction_map[name]
            weight_original = profile.weights[name]

            if result.available:
                raw_score = result.score
                weight_applied = (
                    applied_weights[name] if mode != "neutral_fill" else weight_original
                )
            elif mode == "neutral_fill":
                raw_score = self.neutral_fill_value
                weight_applied = weight_original
            else:
                raw_score = None
                weight_applied = applied_weights.get(name, 0.0)

            weighted_contribution = (raw_score or 0.0) * weight_applied
            final_score += weighted_contribution

            components.append(
                ComponentBreakdown(
                    name=name,
                    available=result.available,
                    raw_score=raw_score,
                    weight_original=weight_original,
                    weight_applied=weight_applied,
                    weighted_contribution=weighted_contribution,
                    notes=result.notes,
                )
            )

        final_score = max(0.0, min(1.0, final_score))

        return ScoringResult(
            candidate_id=candidate_id,
            job_id=job_id,
            role_profile=profile.name,
            final_score=final_score,
            status="scored",
            components=components,
            missing_data_mode=mode,
            components_missing=components_missing,
            weights_renormalized=weights_renormalized,
            warnings=warnings,
        )
