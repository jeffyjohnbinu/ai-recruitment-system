"""
model_optimizer.py
-------------------
Optimizes AI models to reduce false rejections in screening.

Analyzes scoring patterns and provides recommendations for
model improvements that reduce false rejections while
maintaining or improving overall accuracy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from utils.logger import get_logger

logger = get_logger("day30_screening_testing.model_optimizer")


@dataclass
class FalseRejectionCase:
    """A case where a candidate was incorrectly rejected."""

    case_id: str
    candidate_id: str
    score: float
    recommendation: str
    ground_truth: str
    reason: str
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    mitigating_factors: List[str] = field(default_factory=list)


@dataclass
class ModelOptimizationReport:
    """Report of model optimization analysis."""

    total_cases: int
    false_rejections: int
    false_rejection_rate: float
    top_false_rejection_reasons: List[Dict[str, Any]]
    optimization_recommendations: List[Dict[str, Any]]
    dimension_analysis: Dict[str, Dict[str, float]]
    low_confidence_cases: List[FalseRejectionCase]
    high_confidence_misclassifications: List[FalseRejectionCase]
    summary: str


class ModelOptimizer:
    """
    Optimizes AI models to reduce false rejections in screening.

    Analyzes scoring patterns and provides recommendations for
    model improvements that reduce false rejections while
    maintaining or improving overall accuracy.

    Usage:
        optimizer = ModelOptimizer()
        report = optimizer.analyze_false_rejections(
            cases=[...],
            ground_truth=[...],
        )
    """

    def __init__(
        self,
        low_confidence_threshold: float = 0.5,
        high_confidence_threshold: float = 0.9,
    ) -> None:
        """
        Args:
            low_confidence_threshold: Scores below this are "low confidence".
            high_confidence_threshold: Scores above this are "high confidence".
        """
        self.low_confidence_threshold = low_confidence_threshold
        self.high_confidence_threshold = high_confidence_threshold
        self._false_rejections: List[FalseRejectionCase] = []

    def analyze_false_rejections(
        self,
        cases: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
    ) -> ModelOptimizationReport:
        """
        Analyze false rejections in screening results.

        Args:
            cases: List of scored cases with:
                   - case_id, candidate_id, score, recommendation,
                     dimension_scores, reason
            ground_truth: List of ground truth labels with:
                          - case_id, expected_recommendation

        Returns:
            ModelOptimizationReport with analysis and recommendations.
        """
        logger.info("Analyzing false rejections: %d cases", len(cases))

        # Match cases with ground truth
        case_map = {c["case_id"]: c for c in cases}
        gt_map = {g["case_id"]: g for g in ground_truth}

        false_rejections = []
        misclassified_cases = []

        for case_id, case in case_map.items():
            if case_id not in gt_map:
                continue

            gt = gt_map[case_id]
            expected_rec = gt.get("expected_recommendation", "unknown")

            # Check for false rejection
            if case["recommendation"] == "reject" and expected_rec != "reject":
                fr = FalseRejectionCase(
                    case_id=case_id,
                    candidate_id=case.get("candidate_id", ""),
                    score=case.get("score", 0.0),
                    recommendation=case["recommendation"],
                    ground_truth=expected_rec,
                    reason=case.get("reason", ""),
                    dimension_scores=case.get("dimension_scores", {}),
                    mitigating_factors=case.get("mitigating_factors", []),
                )
                false_rejections.append(fr)
                self._false_rejections.append(fr)

            # Track all misclassifications
            if case["recommendation"] != expected_rec:
                misclassified_cases.append(fr)

        # Analyze false rejection reasons
        reason_analysis = self._analyze_reasons(false_rejections)

        # Analyze dimension patterns
        dimension_analysis = self._analyze_dimensions(false_rejections)

        # Identify low confidence cases
        low_confidence = [fr for fr in false_rejections if fr.score < self.low_confidence_threshold]

        # Identify high confidence misclassifications
        high_confidence = [
            fr for fr in false_rejections if fr.score >= self.high_confidence_threshold
        ]

        # Generate recommendations
        recommendations = self._generate_recommendations(
            false_rejections,
            reason_analysis,
            dimension_analysis,
            low_confidence,
            high_confidence,
        )

        # Generate summary
        summary = self._generate_summary(
            len(cases), false_rejections, reason_analysis, recommendations
        )

        report = ModelOptimizationReport(
            total_cases=len(cases),
            false_rejections=len(false_rejections),
            false_rejection_rate=len(false_rejections) / len(cases) if cases else 0.0,
            top_false_rejection_reasons=reason_analysis,
            optimization_recommendations=recommendations,
            dimension_analysis=dimension_analysis,
            low_confidence_cases=low_confidence,
            high_confidence_misclassifications=high_confidence,
            summary=summary,
        )

        logger.info(
            "Model optimization analysis complete: "
            "false_rejection_rate=%.2f%% recommendations=%d",
            report.false_rejection_rate * 100,
            len(recommendations),
        )

        return report

    def identify_mitigating_factors(
        self,
        case: Dict[str, Any],
    ) -> List[str]:
        """
        Identify factors that mitigate false rejections.

        Args:
            case: The case to analyze.

        Returns:
            List of mitigating factors.
        """
        factors = []

        # Check dimension scores
        dim_scores = case.get("dimension_scores", {})
        for dim, score in dim_scores.items():
            if score >= 0.8:
                factors.append(f"Strong {dim} score ({score:.2f})")

        # Check confidence level
        score = case.get("score", 0.0)
        if score >= 0.7 and score < 0.75:
            factors.append("Score near threshold boundary")

        # Check specific question failures
        reason = case.get("reason", "")
        if "experience" in reason.lower():
            factors.append("Experience requirement may be too strict")
        if "skill" in reason.lower():
            factors.append("Skill matching may need tuning")

        return factors

    def optimize_scoring_weights(
        self,
        dimension_analysis: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """
        Optimize dimension weights based on analysis.

        Args:
            dimension_analysis: Dimension-wise analysis from analyze_false_rejections.

        Returns:
            Optimized dimension weights.
        """
        # Default weights
        weights = {
            "clarity": 0.15,
            "relevance": 0.40,
            "completeness": 0.30,
            "consistency": 0.15,
        }

        # Adjust weights based on analysis
        for dim, analysis in dimension_analysis.items():
            if dim not in weights:
                continue

            # Increase weight for dimensions with high false rejection impact
            if analysis.get("false_rejection_impact", 0) > 0.5:
                weights[dim] = min(weights[dim] + 0.05, 0.30)

            # Decrease weight for dimensions with low correlation
            if analysis.get("correlation_with_truth", 0) < 0.3:
                weights[dim] = max(weights[dim] - 0.05, 0.05)

        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def generate_adjustment_rules(
        self,
        report: ModelOptimizationReport,
    ) -> List[Dict[str, Any]]:
        """
        Generate adjustment rules to reduce false rejections.

        Args:
            report: Model optimization report.

        Returns:
            List of adjustment rules.
        """
        rules = []

        # Rule 1: Increase threshold for low confidence cases
        if report.low_confidence_cases:
            rules.append(
                {
                    "type": "threshold_adjustment",
                    "description": "Increase hold threshold for scores near boundary",
                    "condition": "score >= 0.60 AND score < 0.70",
                    "action": "raise threshold by 0.05",
                    "priority": "medium",
                }
            )

        # Rule 2: Add human review for specific dimensions
        for dim, analysis in report.dimension_analysis.items():
            if analysis.get("false_rejection_impact", 0) > 0.6:
                rules.append(
                    {
                        "type": "human_review",
                        "description": f"Add human review for low {dim} scores",
                        "condition": f"{dim}_score < 0.5",
                        "action": "escalate to human reviewer",
                        "priority": "high",
                    }
                )

        # Rule 3: Exception handling for mitigating factors
        rules.append(
            {
                "type": "exception_handling",
                "description": "Add exception handling for mitigating factors",
                "condition": "mitigating_factors detected",
                "action": "review before rejection",
                "priority": "medium",
            }
        )

        return rules

    def _analyze_reasons(
        self,
        false_rejections: List[FalseRejectionCase],
    ) -> List[Dict[str, Any]]:
        """Analyze reasons for false rejections."""
        reason_counts: Dict[str, int] = {}

        for fr in false_rejections:
            reason = fr.reason or "unknown"
            # Normalize reason
            if "experience" in reason.lower():
                reason = "experience_mismatch"
            elif "skill" in reason.lower():
                reason = "skill_mismatch"
            elif "score" in reason.lower():
                reason = "low_score"
            elif "not_answered" in reason.lower():
                reason = "unanswered"

            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        # Sort by frequency and return top reasons
        sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
        return [
            {
                "reason": reason,
                "count": count,
                "percentage": count / len(false_rejections) if false_rejections else 0,
            }
            for reason, count in sorted_reasons[:5]
        ]

    def _analyze_dimensions(
        self,
        false_rejections: List[FalseRejectionCase],
    ) -> Dict[str, Dict[str, float]]:
        """Analyze dimension scores in false rejections."""
        dim_stats: Dict[str, Dict[str, float]] = {}

        for fr in false_rejections:
            for dim, score in fr.dimension_scores.items():
                if dim not in dim_stats:
                    dim_stats[dim] = {"total": 0, "sum": 0, "low_count": 0}
                dim_stats[dim]["total"] += 1
                dim_stats[dim]["sum"] += score
                if score < 0.5:
                    dim_stats[dim]["low_count"] += 1

        # Calculate averages and false rejection impact
        result = {}
        for dim, stats in dim_stats.items():
            avg = stats["sum"] / stats["total"] if stats["total"] > 0 else 0
            low_pct = stats["low_count"] / stats["total"] if stats["total"] > 0 else 0
            result[dim] = {
                "average_score": avg,
                "low_score_percentage": low_pct,
                "false_rejection_impact": low_pct,
                "correlation_with_truth": 1 - low_pct,
            }

        return result

    def _generate_recommendations(
        self,
        false_rejections: List[FalseRejectionCase],
        reason_analysis: List[Dict[str, Any]],
        dimension_analysis: Dict[str, Dict[str, float]],
        low_confidence: List[FalseRejectionCase],
        high_confidence: List[FalseRejectionCase],
    ) -> List[Dict[str, Any]]:
        """Generate optimization recommendations."""
        recommendations = []

        # Threshold adjustments
        if reason_analysis:
            top_reason = reason_analysis[0]["reason"]
            recommendations.append(
                {
                    "type": "threshold_adjustment",
                    "description": f"Adjust thresholds for {top_reason} cases",
                    "detail": f"Top false rejection reason: {top_reason} "
                    f"({reason_analysis[0]['percentage']:.1%} of cases)",
                    "priority": "high",
                }
            )

        # Dimension weight tuning
        for dim, analysis in dimension_analysis.items():
            if analysis.get("false_rejection_impact", 0) > 0.5:
                recommendations.append(
                    {
                        "type": "weight_tuning",
                        "description": f"Reduce weight for {dim} dimension",
                        "detail": f"Current avg score: {analysis['average_score']:.2f}, "
                        f"low score rate: {analysis['low_score_percentage']:.1%}",
                        "priority": "medium",
                    }
                )

        # Low confidence handling
        if low_confidence:
            recommendations.append(
                {
                    "type": "confidence_handling",
                    "description": f"Add human review for {len(low_confidence)} low-confidence cases",
                    "detail": f"Cases with score < {self.low_confidence_threshold}",
                    "priority": "high",
                }
            )

        # High confidence misclassifications
        if high_confidence:
            recommendations.append(
                {
                    "type": "model_improvement",
                    "description": f"Investigate {len(high_confidence)} high-confidence misclassifications",
                    "detail": "These cases need model retraining or rule adjustment",
                    "priority": "critical",
                }
            )

        return recommendations

    def _generate_summary(
        self,
        total_cases: int,
        false_rejections: List[FalseRejectionCase],
        reason_analysis: List[Dict[str, Any]],
        recommendations: List[Dict[str, Any]],
    ) -> str:
        """Generate human-readable summary."""
        frr = len(false_rejections) / total_cases if total_cases > 0 else 0
        return (
            f"Analysis of {total_cases} cases found {len(false_rejections)} "
            f"false rejections ({frr:.1%}). "
            f"Top reason: {reason_analysis[0]['reason'] if reason_analysis else 'N/A'}. "
            f"Generated {len(recommendations)} optimization recommendations."
        )

    def export_report(self, report: ModelOptimizationReport, filepath: str) -> None:
        """Export optimization report to JSON."""
        data = {
            "total_cases": report.total_cases,
            "false_rejections": report.false_rejections,
            "false_rejection_rate": report.false_rejection_rate,
            "top_false_rejection_reasons": report.top_false_rejection_reasons,
            "optimization_recommendations": report.optimization_recommendations,
            "dimension_analysis": report.dimension_analysis,
            "low_confidence_cases": [
                {"case_id": fr.case_id, "score": fr.score, "reason": fr.reason}
                for fr in report.low_confidence_cases
            ],
            "high_confidence_misclassifications": [
                {"case_id": fr.case_id, "score": fr.score, "reason": fr.reason}
                for fr in report.high_confidence_misclassifications
            ],
            "summary": report.summary,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Optimization report exported to %s", filepath)
