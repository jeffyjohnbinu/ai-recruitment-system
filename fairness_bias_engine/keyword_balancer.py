"""
keyword_balancer.py
--------------------
Reduces over-dependence on raw keyword matching in the final score.

Problem this solves:
    A candidate who pads a resume with every keyword from the job
    description (skill-stuffing) can out-score a genuinely stronger
    candidate who wrote naturally, if the "keyword_match" component of
    the Day 13 ATS Scoring Engine is left uncapped. Keyword count is a
    weak, gameable proxy for actual fit; semantic similarity (Day 12)
    and experience relevance (Day 10) are harder to game and more
    predictive.

Approach:
    1. Apply a diminishing-returns curve to the keyword-match component
       itself, so the Nth extra matched keyword contributes less than
       the first (`sqrt` dampening beyond a configurable threshold).
    2. Cap the *share* of the final weighted score that the keyword
       component is allowed to contribute (default 35%), redistributing
       any excess weight proportionally across the remaining components
       (semantic similarity, experience relevance, education relevance,
       etc.) rather than discarding it.

This module operates on the same `component_scores` shape produced by
the Day 13 ATS Scoring Engine's WeightProfile (a dict of
component_name -> (raw_score, weight)), so it can be dropped into the
pipeline between Day 13 scoring and Day 14 ranking without altering
either module's schema.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

DEFAULT_MAX_KEYWORD_SHARE = 0.35
DEFAULT_DIMINISHING_THRESHOLD = 8  # matched keywords beyond this count get dampened
KEYWORD_COMPONENT_ALIASES = {"keyword_match", "keyword_score", "keyword_overlap"}


@dataclass
class KeywordBalanceReport:
    original_components: Dict[str, Tuple[float, float]]
    adjusted_components: Dict[str, Tuple[float, float]]
    keyword_component_name: str | None
    dampening_applied: bool
    weight_redistributed: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "original_components": {k: list(v) for k, v in self.original_components.items()},
            "adjusted_components": {k: list(v) for k, v in self.adjusted_components.items()},
            "keyword_component_name": self.keyword_component_name,
            "dampening_applied": self.dampening_applied,
            "weight_redistributed": self.weight_redistributed,
            "notes": self.notes,
        }


class KeywordDependencyReducer:
    """
    Caps and dampens the influence of keyword-matching on the final score.

    Parameters:
        max_keyword_share: the maximum fraction (0-1) of total weight the
            keyword component may retain after redistribution.
        diminishing_threshold: number of matched keywords after which
            additional matches yield sqrt-dampened (not linear) benefit.
    """

    def __init__(
        self,
        max_keyword_share: float = DEFAULT_MAX_KEYWORD_SHARE,
        diminishing_threshold: int = DEFAULT_DIMINISHING_THRESHOLD,
    ):
        self.max_keyword_share = max_keyword_share
        self.diminishing_threshold = diminishing_threshold

    def apply_diminishing_returns(
        self, matched_keyword_count: int, total_keyword_count: int
    ) -> float:
        """
        Returns a dampened keyword raw score in [0, 1]. Below the threshold,
        behaves like a plain ratio. Beyond it, extra matches are compressed
        with a sqrt curve so stuffing keywords has rapidly shrinking payoff.
        """
        if total_keyword_count <= 0:
            return 0.0
        if matched_keyword_count <= self.diminishing_threshold:
            return max(0.0, min(1.0, matched_keyword_count / total_keyword_count))

        base = self.diminishing_threshold
        extra = matched_keyword_count - base
        dampened_extra = math.sqrt(extra)
        effective_count = base + dampened_extra
        return max(0.0, min(1.0, effective_count / total_keyword_count))

    def rebalance(self, component_scores: Dict[str, Tuple[float, float]]) -> KeywordBalanceReport:
        """
        component_scores: {component_name: (raw_score_0_to_1, weight_0_to_1)}
        Weights are expected to sum to ~1.0 (as produced by Day 13's WeightProfile).
        """
        notes: List[str] = []
        keyword_name = next(
            (name for name in component_scores if name.lower() in KEYWORD_COMPONENT_ALIASES),
            None,
        )

        adjusted = dict(component_scores)

        if keyword_name is None:
            notes.append("no keyword-matching component found; nothing to rebalance")
            return KeywordBalanceReport(
                original_components=dict(component_scores),
                adjusted_components=adjusted,
                keyword_component_name=None,
                dampening_applied=False,
                weight_redistributed=0.0,
                notes=notes,
            )

        raw_score, weight = component_scores[keyword_name]
        redistributed = 0.0
        dampening_applied = False

        if weight > self.max_keyword_share:
            redistributed = weight - self.max_keyword_share
            weight = self.max_keyword_share
            dampening_applied = True
            notes.append(
                f"capped '{keyword_name}' weight at {self.max_keyword_share:.2f} "
                f"(redistributed {redistributed:.3f} to other components)"
            )

        adjusted[keyword_name] = (raw_score, weight)

        other_names = [n for n in component_scores if n != keyword_name]
        other_weight_total = sum(component_scores[n][1] for n in other_names)

        if redistributed > 0 and other_weight_total > 0:
            for name in other_names:
                r, w = component_scores[name]
                share = w / other_weight_total
                adjusted[name] = (r, w + redistributed * share)
        elif redistributed > 0:
            notes.append("no other components available to absorb redistributed weight")

        return KeywordBalanceReport(
            original_components=dict(component_scores),
            adjusted_components=adjusted,
            keyword_component_name=keyword_name,
            dampening_applied=dampening_applied,
            weight_redistributed=redistributed,
            notes=notes,
        )

    @staticmethod
    def weighted_total(component_scores: Dict[str, Tuple[float, float]]) -> float:
        return sum(raw * weight for raw, weight in component_scores.values())
