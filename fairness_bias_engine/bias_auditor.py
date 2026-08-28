"""
bias_auditor.py
----------------
Evaluates bias indicators across a scored/ranked candidate pool.

This module does NOT infer demographic attributes on its own -- doing so
from a resume would itself be a bias risk. Instead, it accepts optional,
separately-supplied group labels (`demographic_groups`), typically
provided out-of-band by a compliance/HR process for audit purposes only,
never fed into scoring. If no group labels are supplied, the auditor
still reports pool-level score distribution stats but skips disparate
impact analysis and says so explicitly.

Primary method: the four-fifths rule (a standard, widely used adverse
impact screen: https://www.eeoc.gov "Uniform Guidelines"), which flags a
potential concern when the selection rate for any group is less than 80%
of the group with the highest selection rate. This is a screening
heuristic, not a legal conclusion -- the report says so.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

FOUR_FIFTHS_THRESHOLD = 0.8
MIN_GROUP_SIZE_FOR_CONFIDENCE = 5


@dataclass
class GroupStat:
    group_name: str
    group_size: int
    shortlisted_count: int
    selection_rate: float
    mean_score: float
    median_score: float
    low_confidence: bool  # True if group_size < MIN_GROUP_SIZE_FOR_CONFIDENCE

    def to_dict(self) -> Dict[str, object]:
        return {
            "group_name": self.group_name,
            "group_size": self.group_size,
            "shortlisted_count": self.shortlisted_count,
            "selection_rate": round(self.selection_rate, 4),
            "mean_score": round(self.mean_score, 4),
            "median_score": round(self.median_score, 4),
            "low_confidence": self.low_confidence,
        }


@dataclass
class BiasAuditReport:
    dimension: Optional[str]
    group_stats: List[GroupStat] = field(default_factory=list)
    four_fifths_ratio: Optional[float] = None
    flagged: bool = False
    insufficient_data: bool = False
    narrative: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "dimension": self.dimension,
            "group_stats": [g.to_dict() for g in self.group_stats],
            "four_fifths_ratio": (
                round(self.four_fifths_ratio, 4) if self.four_fifths_ratio is not None else None
            ),
            "flagged": self.flagged,
            "insufficient_data": self.insufficient_data,
            "narrative": self.narrative,
        }


class BiasAuditor:
    """
    Computes selection-rate parity (four-fifths rule) and score
    distribution stats per group, for one or more demographic dimensions.
    """

    def audit_dimension(
        self,
        dimension_name: str,
        candidates: List[Dict[str, object]],
    ) -> BiasAuditReport:
        """
        candidates: list of dicts each containing at least:
            - "group": the group label for this dimension (e.g. "gender:female")
            - "score": float final score
            - "shortlisted": bool
        """
        groups: Dict[str, List[Dict[str, object]]] = {}
        for c in candidates:
            group = c.get("group")
            if group is None:
                continue
            groups.setdefault(str(group), []).append(c)

        if not groups:
            return BiasAuditReport(
                dimension=dimension_name,
                insufficient_data=True,
                narrative=(
                    f"No group labels supplied for dimension '{dimension_name}'; "
                    "disparate impact analysis was skipped. Score-based screening "
                    "was not evaluated for bias on this dimension."
                ),
            )

        group_stats: List[GroupStat] = []
        for group_name, members in groups.items():
            size = len(members)
            shortlisted = sum(1 for m in members if m.get("shortlisted"))
            scores = [float(m["score"]) for m in members if "score" in m]
            selection_rate = shortlisted / size if size else 0.0
            mean_score = statistics.fmean(scores) if scores else 0.0
            median_score = statistics.median(scores) if scores else 0.0
            group_stats.append(
                GroupStat(
                    group_name=group_name,
                    group_size=size,
                    shortlisted_count=shortlisted,
                    selection_rate=selection_rate,
                    mean_score=mean_score,
                    median_score=median_score,
                    low_confidence=size < MIN_GROUP_SIZE_FOR_CONFIDENCE,
                )
            )

        insufficient_data = all(g.low_confidence for g in group_stats) or len(group_stats) < 2

        four_fifths_ratio = None
        flagged = False
        if len(group_stats) >= 2:
            rates = [g.selection_rate for g in group_stats]
            max_rate = max(rates)
            min_rate = min(rates)
            if max_rate > 0:
                four_fifths_ratio = min_rate / max_rate
                flagged = four_fifths_ratio < FOUR_FIFTHS_THRESHOLD

        narrative = self._build_narrative(
            dimension_name, group_stats, four_fifths_ratio, flagged, insufficient_data
        )

        return BiasAuditReport(
            dimension=dimension_name,
            group_stats=group_stats,
            four_fifths_ratio=four_fifths_ratio,
            flagged=flagged and not insufficient_data,
            insufficient_data=insufficient_data,
            narrative=narrative,
        )

    @staticmethod
    def _build_narrative(
        dimension: str,
        group_stats: List[GroupStat],
        ratio: Optional[float],
        flagged: bool,
        insufficient_data: bool,
    ) -> str:
        if not group_stats:
            return f"No data available for dimension '{dimension}'."

        parts = [
            f"{g.group_name}: selection rate {g.selection_rate:.1%} (n={g.group_size})"
            for g in group_stats
        ]
        summary = f"Dimension '{dimension}' — " + "; ".join(parts) + "."

        if insufficient_data:
            summary += (
                " Sample size is too small in at least one group for a reliable "
                "disparate-impact read; treat this as informational only."
            )
        elif ratio is not None:
            summary += f" Four-fifths ratio = {ratio:.2f} (threshold {FOUR_FIFTHS_THRESHOLD})."
            if flagged:
                summary += (
                    " This falls below the four-fifths screening threshold and "
                    "warrants closer review of the scoring/shortlisting criteria "
                    "for this dimension. This is a screening heuristic, not a "
                    "legal or causal determination."
                )
            else:
                summary += " No adverse-impact flag under the four-fifths screen."

        return summary
