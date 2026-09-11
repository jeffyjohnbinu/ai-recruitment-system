"""
scoring_threshold_tuner.py
--------------------------
Tunes scoring thresholds to reduce false rejections and improve
screening system accuracy.

Analyzes scoring results against ground truth to find optimal
thresholds for the recommendation classification.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger("day30_screening_testing.scoring_threshold_tuner")


@dataclass
class ThresholdTestResult:
    """Result of testing with a specific threshold configuration."""

    proceed_threshold: float
    hold_threshold: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positives: int
    false_negatives: int
    true_positives: int
    true_negatives: int

    @property
    def harmonic_mean(self) -> float:
        """Harmonic mean of precision and recall (F1 score)."""
        return self.f1_score


@dataclass
class TuningReport:
    """Report of threshold tuning results."""

    best_threshold: Tuple[float, float]
    best_result: ThresholdTestResult
    all_results: List[ThresholdTestResult] = field(default_factory=list)
    original_accuracy: float = 0.0
    original_false_rejection_rate: float = 0.0
    optimized_false_rejection_rate: float = 0.0
    false_rejection_reduction: float = 0.0
    improvement_summary: str = ""


class ScoringThresholdTuner:
    """
    Tunes scoring thresholds to optimize recommendation accuracy.

    Uses ground truth labels to find optimal thresholds that minimize
    false positives and false negatives, particularly minimizing
    false rejections.

    Usage:
        tuner = ScoringThresholdTuner()
        report = tuner.tune_thresholds(
            scores=[0.82, 0.45, 0.18, ...],
            true_recommendations=["proceed", "hold", "reject", ...],
        )
        print(report.best_threshold)
    """

    # Default threshold grid for tuning
    DEFAULT_THRESHOLD_RANGE = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    DEFAULT_HOLD_RANGE = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]

    def __init__(
        self,
        threshold_range: Optional[List[float]] = None,
        hold_range: Optional[List[float]] = None,
    ) -> None:
        """
        Args:
            threshold_range: Range of proceed threshold values to test.
            hold_range: Range of hold threshold values to test.
        """
        self.threshold_range = threshold_range or self.DEFAULT_THRESHOLD_RANGE
        self.hold_range = hold_range or self.DEFAULT_HOLD_RANGE

    def tune_thresholds(
        self,
        scores: List[float],
        true_recommendations: List[str],
        current_thresholds: Optional[Tuple[float, float]] = None,
    ) -> TuningReport:
        """
        Tune thresholds to optimize recommendation accuracy.

        Args:
            scores: List of normalized scores (0-1) from scoring engine.
            true_recommendations: List of ground truth recommendations.
                                Each should be "proceed", "hold", or "reject".
            current_thresholds: Current (proceed_threshold, hold_threshold).

        Returns:
            TuningReport with best thresholds and performance metrics.
        """
        logger.info(
            "Tuning thresholds: %d samples, current=%s",
            len(scores),
            current_thresholds,
        )

        # Calculate original performance
        original_result = self._evaluate_thresholds(
            scores,
            true_recommendations,
            current_thresholds or (0.75, 0.45),
        )
        original_accuracy = original_result.accuracy
        original_fpr = self._calculate_false_rejection_rate(
            scores,
            true_recommendations,
            current_thresholds or (0.75, 0.45),
        )

        # Grid search for optimal thresholds
        all_results: List[ThresholdTestResult] = []
        best_result: Optional[ThresholdTestResult] = None
        best_thresholds: Tuple[float, float] = (0.75, 0.45)

        for proceed_thresh in self.threshold_range:
            for hold_thresh in self.hold_range:
                if hold_thresh >= proceed_thresh:
                    continue

                result = self._evaluate_thresholds(
                    scores,
                    true_recommendations,
                    (proceed_thresh, hold_thresh),
                )
                all_results.append(result)

                # Select best: prefer higher F1, then lower FPR
                if best_result is None:
                    best_result = result
                    best_thresholds = (proceed_thresh, hold_thresh)
                elif self._is_better_result(result, best_result):
                    best_result = result
                    best_thresholds = (proceed_thresh, hold_thresh)

        optimized_fpr = self._calculate_false_rejection_rate(
            scores, true_recommendations, best_thresholds
        )
        false_rejection_reduction = (
            (original_fpr - optimized_fpr) / original_fpr * 100 if original_fpr > 0 else 0.0
        )

        # Generate improvement summary
        summary = self._generate_summary(
            original_accuracy, original_fpr, best_result, optimized_fpr, false_rejection_reduction
        )

        report = TuningReport(
            best_threshold=best_thresholds,
            best_result=best_result,
            all_results=all_results,
            original_accuracy=original_accuracy,
            original_false_rejection_rate=original_fpr,
            optimized_false_rejection_rate=optimized_fpr,
            false_rejection_reduction=false_rejection_reduction,
            improvement_summary=summary,
        )

        logger.info(
            "Tuning complete: original_fpr=%.2f%% optimized_fpr=%.2f%% reduction=%.1f%%",
            original_fpr * 100,
            optimized_fpr * 100,
            false_rejection_reduction,
        )

        return report

    def evaluate_threshold(
        self,
        scores: List[float],
        true_recommendations: List[str],
        thresholds: Tuple[float, float],
    ) -> ThresholdTestResult:
        """Evaluate performance at specific thresholds."""
        return self._evaluate_thresholds(scores, true_recommendations, thresholds)

    def _evaluate_thresholds(
        self,
        scores: List[float],
        true_recommendations: List[str],
        thresholds: Tuple[float, float],
    ) -> ThresholdTestResult:
        """Evaluate recommendation accuracy at given thresholds."""
        proceed_thresh, hold_thresh = thresholds

        # Convert recommendations to binary (shortlisted = proceed)
        predictions = []
        true_labels = []

        for score, true_rec in zip(scores, true_recommendations):
            predicted = self._score_to_recommendation(score, proceed_thresh, hold_thresh)
            predictions.append(predicted)
            true_labels.append(true_rec)

        # Calculate confusion matrix
        tp = sum(1 for p, t in zip(predictions, true_labels) if p == "proceed" and t == "proceed")
        fp = sum(1 for p, t in zip(predictions, true_labels) if p == "proceed" and t != "proceed")
        tn = sum(1 for p, t in zip(predictions, true_labels) if p != "proceed" and t != "proceed")
        fn = sum(1 for p, t in zip(predictions, true_labels) if p != "proceed" and t == "proceed")

        total = len(scores)
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return ThresholdTestResult(
            proceed_threshold=proceed_thresh,
            hold_threshold=hold_thresh,
            accuracy=round(accuracy, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1, 4),
            false_positives=fp,
            false_negatives=fn,
            true_positives=tp,
            true_negatives=tn,
        )

    def _score_to_recommendation(
        self,
        score: float,
        proceed_threshold: float,
        hold_threshold: float,
    ) -> str:
        """Convert a score to a recommendation using thresholds."""
        if score >= proceed_threshold:
            return "proceed"
        elif score >= hold_threshold:
            return "hold"
        else:
            return "reject"

    def _calculate_false_rejection_rate(
        self,
        scores: List[float],
        true_recommendations: List[str],
        thresholds: Tuple[float, float],
    ) -> float:
        """Calculate the false rejection rate (false negatives for proceed)."""
        proceed_thresh, hold_thresh = thresholds

        false_rejections = 0
        total_rejections = 0

        for score, true_rec in zip(scores, true_recommendations):
            predicted = self._score_to_recommendation(score, proceed_thresh, hold_thresh)
            if predicted == "reject":
                total_rejections += 1
                if true_rec == "proceed":
                    false_rejections += 1

        return false_rejections / total_rejections if total_rejections > 0 else 0.0

    def _is_better_result(
        self,
        candidate: ThresholdTestResult,
        current_best: ThresholdTestResult,
    ) -> bool:
        """Determine if candidate result is better than current best."""
        # Primary: minimize F1 score difference but prioritize lower FPR
        # Secondary: minimize false negatives (rejecting good candidates)
        if candidate.f1_score > current_best.f1_score + 0.001:
            return True
        if candidate.f1_score < current_best.f1_score - 0.001:
            return False
        # Equal F1: prefer fewer false negatives
        return candidate.false_negatives < current_best.false_negatives

    def _generate_summary(
        self,
        original_accuracy: float,
        original_fpr: float,
        best_result: ThresholdTestResult,
        optimized_fpr: float,
        false_rejection_reduction: float,
    ) -> str:
        """Generate human-readable improvement summary."""
        return (
            f"Threshold tuning optimized accuracy from {original_accuracy:.1%} to "
            f"{best_result.accuracy:.1%} and reduced the false rejection rate "
            f"from {original_fpr:.1%} to {optimized_fpr:.1%} "
            f"(a {false_rejection_reduction:.1f}% relative reduction). "
            f"Best thresholds: proceed >= {best_result.proceed_threshold}, "
            f"hold >= {best_result.hold_threshold}."
        )

    def export_report(self, report: TuningReport, filepath: str) -> None:
        """Export tuning report to JSON."""
        data = {
            "best_threshold": report.best_threshold,
            "best_result": report.best_result.__dict__,
            "all_results": [r.__dict__ for r in report.all_results],
            "original_accuracy": report.original_accuracy,
            "original_false_rejection_rate": report.original_false_rejection_rate,
            "optimized_false_rejection_rate": report.optimized_false_rejection_rate,
            "false_rejection_reduction": report.false_rejection_reduction,
            "improvement_summary": report.improvement_summary,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Tuning report exported to %s", filepath)
