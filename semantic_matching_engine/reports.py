"""
Matching Accuracy Report
--------------------------
Builds the "Matching accuracy report" deliverable: given a set of
(candidate_id, job_id, overall_score, predicted_is_match, ground_truth_is_match)
rows -- typically produced by running `SemanticMatchingEngine.match_many()`
against a labeled validation set spanning multiple job types -- compute
precision, recall, F1, accuracy, and a confusion matrix, plus a per-job-type
breakdown so uneven performance across job families is visible rather than
averaged away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List


@dataclass
class ValidationRow:
    candidate_id: str
    job_id: str
    overall_score: float
    predicted_is_match: bool
    actual_is_match: bool
    job_type: str = "unspecified"


@dataclass
class AccuracyReport:
    generated_at: str
    total_pairs: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    by_job_type: Dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "total_pairs": self.total_pairs,
            "confusion_matrix": {
                "true_positives": self.true_positives,
                "false_positives": self.false_positives,
                "true_negatives": self.true_negatives,
                "false_negatives": self.false_negatives,
            },
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "accuracy": round(self.accuracy, 4),
            "by_job_type": self.by_job_type,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Semantic Matching Accuracy Report",
            "",
            f"Generated: {self.generated_at}",
            f"Validation pairs: {self.total_pairs}",
            "",
            "## Overall",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Precision | {self.precision:.3f} |",
            f"| Recall | {self.recall:.3f} |",
            f"| F1 score | {self.f1_score:.3f} |",
            f"| Accuracy | {self.accuracy:.3f} |",
            "",
            "## Confusion matrix",
            "",
            "| | Predicted match | Predicted no-match |",
            "|---|---|---|",
            f"| **Actual match** | {self.true_positives} (TP) | {self.false_negatives} (FN) |",
            f"| **Actual no-match** | {self.false_positives} (FP) | {self.true_negatives} (TN) |",
            "",
        ]
        if self.by_job_type:
            lines += [
                "## By job type",
                "",
                "| Job type | Pairs | Precision | Recall | F1 |",
                "|---|---|---|---|---|",
            ]
            for job_type, stats in self.by_job_type.items():
                lines.append(
                    f"| {job_type} | {stats['total_pairs']} | {stats['precision']:.3f} | "
                    f"{stats['recall']:.3f} | {stats['f1_score']:.3f} |"
                )
            lines.append("")
        return "\n".join(lines)


def _compute_confusion(rows: List[ValidationRow]) -> Dict[str, int]:
    tp = fp = tn = fn = 0
    for row in rows:
        if row.predicted_is_match and row.actual_is_match:
            tp += 1
        elif row.predicted_is_match and not row.actual_is_match:
            fp += 1
        elif not row.predicted_is_match and not row.actual_is_match:
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _prf1_accuracy(counts: Dict[str, int]) -> Dict[str, float]:
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    return {"precision": precision, "recall": recall, "f1_score": f1, "accuracy": accuracy}


def generate_accuracy_report(rows: List[ValidationRow]) -> AccuracyReport:
    if not rows:
        raise ValueError("Cannot generate an accuracy report from an empty validation set.")

    overall_counts = _compute_confusion(rows)
    overall_metrics = _prf1_accuracy(overall_counts)

    by_job_type: Dict[str, dict] = {}
    job_types = sorted({row.job_type for row in rows})
    for job_type in job_types:
        subset = [r for r in rows if r.job_type == job_type]
        counts = _compute_confusion(subset)
        metrics = _prf1_accuracy(counts)
        by_job_type[job_type] = {
            "total_pairs": len(subset),
            **{k: round(v, 4) for k, v in metrics.items()},
        }

    return AccuracyReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        total_pairs=len(rows),
        true_positives=overall_counts["tp"],
        false_positives=overall_counts["fp"],
        true_negatives=overall_counts["tn"],
        false_negatives=overall_counts["fn"],
        precision=overall_metrics["precision"],
        recall=overall_metrics["recall"],
        f1_score=overall_metrics["f1_score"],
        accuracy=overall_metrics["accuracy"],
        by_job_type=by_job_type,
    )
