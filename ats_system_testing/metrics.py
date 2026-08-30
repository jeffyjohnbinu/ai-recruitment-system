"""
metrics.py
-----------
Computes accuracy metrics for a batch of CaseResult objects, comparing
the automated pipeline's shortlist decision against manually-reviewed
ground truth. Also produces per-segment (role_type x seniority)
breakdowns and a structured mismatch log, since an aggregate accuracy
number hides exactly the failure modes Day 17 is meant to surface.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

from .harness import CaseResult


@dataclass
class ConfusionCounts:
    true_positive: int = 0  # AI shortlisted, human agreed
    false_positive: int = 0  # AI shortlisted, human said no
    true_negative: int = 0  # AI rejected, human agreed
    false_negative: int = 0  # AI rejected, human said shortlist

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return round(self.true_positive / denom, 4) if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return round(self.true_positive / denom, 4) if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return round(2 * p * r / (p + r), 4) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        total = self.true_positive + self.false_positive + self.true_negative + self.false_negative
        correct = self.true_positive + self.true_negative
        return round(correct / total, 4) if total else 0.0

    @property
    def total(self) -> int:
        return self.true_positive + self.false_positive + self.true_negative + self.false_negative

    def to_dict(self) -> Dict[str, float]:
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "true_negative": self.true_negative,
            "false_negative": self.false_negative,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "total_cases": self.total,
        }


@dataclass
class MismatchEntry:
    case_id: str
    role_type: str
    seniority: str
    ai_shortlisted: bool
    ai_recommendation: str
    human_shortlisted: bool
    human_recommendation: str
    mismatch_type: str  # "false_positive" | "false_negative" | "recommendation_only"
    notes: str

    def to_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


@dataclass
class MetricsReport:
    overall: ConfusionCounts
    by_role_type: Dict[str, ConfusionCounts]
    by_seniority: Dict[str, ConfusionCounts]
    mismatches: List[MismatchEntry] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "overall": self.overall.to_dict(),
            "by_role_type": {k: v.to_dict() for k, v in self.by_role_type.items()},
            "by_seniority": {k: v.to_dict() for k, v in self.by_seniority.items()},
            "mismatches": [m.to_dict() for m in self.mismatches],
        }


def _update_counts(counts: ConfusionCounts, ai_shortlisted: bool, human_shortlisted: bool) -> None:
    if ai_shortlisted and human_shortlisted:
        counts.true_positive += 1
    elif ai_shortlisted and not human_shortlisted:
        counts.false_positive += 1
    elif not ai_shortlisted and not human_shortlisted:
        counts.true_negative += 1
    else:
        counts.false_negative += 1


def compute_metrics(results: List[CaseResult]) -> MetricsReport:
    overall = ConfusionCounts()
    by_role_type: Dict[str, ConfusionCounts] = defaultdict(ConfusionCounts)
    by_seniority: Dict[str, ConfusionCounts] = defaultdict(ConfusionCounts)
    mismatches: List[MismatchEntry] = []

    for r in results:
        ai_shortlisted = r.pipeline_output.shortlisted
        human_shortlisted = r.test_case.ground_truth.shortlisted

        _update_counts(overall, ai_shortlisted, human_shortlisted)
        _update_counts(by_role_type[r.test_case.role_type], ai_shortlisted, human_shortlisted)
        _update_counts(by_seniority[r.test_case.seniority], ai_shortlisted, human_shortlisted)

        recommendation_mismatch = (
            r.pipeline_output.recommendation != r.test_case.ground_truth.recommendation
        )

        if ai_shortlisted != human_shortlisted or recommendation_mismatch:
            if ai_shortlisted and not human_shortlisted:
                mismatch_type = "false_positive"
            elif not ai_shortlisted and human_shortlisted:
                mismatch_type = "false_negative"
            else:
                mismatch_type = "recommendation_only"

            mismatches.append(
                MismatchEntry(
                    case_id=r.test_case.case_id,
                    role_type=r.test_case.role_type,
                    seniority=r.test_case.seniority,
                    ai_shortlisted=ai_shortlisted,
                    ai_recommendation=r.pipeline_output.recommendation,
                    human_shortlisted=human_shortlisted,
                    human_recommendation=r.test_case.ground_truth.recommendation,
                    notes=r.test_case.ground_truth.reviewer_notes,
                    mismatch_type=mismatch_type,
                )
            )

    return MetricsReport(
        overall=overall,
        by_role_type=dict(by_role_type),
        by_seniority=dict(by_seniority),
        mismatches=mismatches,
    )
