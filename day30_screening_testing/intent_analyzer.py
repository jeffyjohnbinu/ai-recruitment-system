"""
intent_analyzer.py
------------------
Analyzes and improves intent detection for AI screening conversations.

Validates intent classification accuracy against ground truth labels
and provides recommendations for improving intent detection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger("day30_screening_testing.intent_analyzer")


# Intent categories used in screening conversations
INTENT_CATEGORIES = [
    "answer",
    "boolean",
    "numeric",
    "no_response",
    "off_topic",
    "objection",
    "redirect",
    "clarification_request",
    "confusion",
    "silence",
    "followup",
]


@dataclass
class IntentPrediction:
    """A single intent prediction with ground truth for validation."""

    answer_text: str
    predicted_intent: str
    true_intent: str
    confidence: float
    is_correct: bool = False
    error_type: Optional[str] = None


@dataclass
class IntentAccuracyReport:
    """Report of intent detection accuracy across a batch of predictions."""

    total_predictions: int
    correct_predictions: int
    accuracy: float
    per_intent_accuracy: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    common_errors: List[Dict[str, Any]] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)


@dataclass
class IntentThresholdConfig:
    """Configuration for intent detection thresholds."""

    confidence_threshold: float = 0.7
    ambiguity_threshold: float = 0.4
    min_answer_length: int = 2
    max_repetition_count: int = 2
    silence_timeout_seconds: float = 3.0


class IntentAnalyzer:
    """
    Analyzes and improves intent detection for screening conversations.

    Validates intent classification against ground truth and provides
    recommendations for improving detection accuracy.

    Usage:
        analyzer = IntentAnalyzer()
        report = analyzer.analyze_intents(
            predictions=[
                {"answer": "Yes", "predicted": "boolean", "true": "boolean"},
                {"answer": "5 years", "predicted": "numeric", "true": "numeric"},
            ]
        )
    """

    def __init__(
        self,
        config: Optional[IntentThresholdConfig] = None,
    ) -> None:
        """
        Args:
            config: Intent detection configuration.
        """
        self.config = config or IntentThresholdConfig()
        self._predictions: List[IntentPrediction] = []

    def analyze_intents(
        self,
        predictions: List[Dict[str, Any]],
    ) -> IntentAccuracyReport:
        """
        Analyze intent predictions against ground truth.

        Args:
            predictions: List of prediction dictionaries with:
                        - answer_text: The answer given
                        - predicted_intent: The predicted intent
                        - true_intent: The ground truth intent
                        - confidence: (optional) Prediction confidence

        Returns:
            IntentAccuracyReport with accuracy metrics and suggestions.
        """
        logger.info("Analyzing %d intent predictions", len(predictions))

        # Validate predictions
        validated = self._validate_predictions(predictions)
        self._predictions = validated

        # Calculate overall accuracy
        correct = sum(1 for p in validated if p.is_correct)
        total = len(validated)
        accuracy = correct / total if total > 0 else 0.0

        # Calculate per-intent accuracy
        per_intent = self._calculate_per_intent_accuracy(validated)

        # Build confusion matrix
        confusion = self._build_confusion_matrix(validated)

        # Identify common errors
        common_errors = self._identify_common_errors(validated)

        # Generate improvement suggestions
        suggestions = self._generate_improvement_suggestions(validated, per_intent, confusion)

        report = IntentAccuracyReport(
            total_predictions=total,
            correct_predictions=correct,
            accuracy=accuracy,
            per_intent_accuracy=per_intent,
            confusion_matrix=confusion,
            common_errors=common_errors,
            improvement_suggestions=suggestions,
        )

        logger.info(
            "Intent analysis complete: accuracy=%.2f%% errors=%d",
            accuracy * 100,
            len(common_errors),
        )

        return report

    def validate_prediction(
        self,
        answer_text: str,
        predicted_intent: str,
        true_intent: str,
        confidence: float = 0.5,
    ) -> IntentPrediction:
        """Validate a single intent prediction."""
        is_correct = predicted_intent == true_intent
        error_type = None if is_correct else self._classify_error(predicted_intent, true_intent)

        prediction = IntentPrediction(
            answer_text=answer_text,
            predicted_intent=predicted_intent,
            true_intent=true_intent,
            confidence=confidence,
            is_correct=is_correct,
            error_type=error_type,
        )

        self._predictions.append(prediction)
        return prediction

    def get_improvement_recommendations(
        self,
        min_confidence: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations for improving intent detection.

        Args:
            min_confidence: Minimum confidence threshold for recommendations.

        Returns:
            List of improvement recommendations.
        """
        if not self._predictions:
            return []

        recommendations = []

        # Analyze low-confidence predictions
        low_confidence = [
            p for p in self._predictions if p.confidence < min_confidence and not p.is_correct
        ]
        if low_confidence:
            recommendations.append(
                {
                    "type": "confidence_threshold",
                    "description": f"Reduce confidence threshold from {self.config.confidence_threshold} to improve recall",
                    "impact": f"May improve accuracy by {len(low_confidence) / len(self._predictions) * 100:.1f}%",
                    "priority": "medium",
                }
            )

        # Analyze common error patterns
        error_types = {}
        for p in self._predictions:
            if p.error_type:
                error_types[p.error_type] = error_types.get(p.error_type, 0) + 1

        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:3]:
            recommendations.append(
                {
                    "type": "error_pattern",
                    "description": f"Address {error_type} errors ({count} occurrences)",
                    "impact": f"Reduce {error_type} errors by 50%",
                    "priority": "high" if count > 5 else "medium",
                }
            )

        # Analyze ambiguous intents
        ambiguous = [p for p in self._predictions if p.confidence < self.config.ambiguity_threshold]
        if ambiguous:
            recommendations.append(
                {
                    "type": "ambiguity_handling",
                    "description": f"Add disambiguation logic for {len(ambiguous)} ambiguous predictions",
                    "impact": "Reduce misclassification rate",
                    "priority": "high",
                }
            )

        return recommendations

    def _validate_predictions(
        self,
        predictions: List[Any],
    ) -> List[IntentPrediction]:
        """Validate a list of predictions."""
        validated = []
        for pred in predictions:
            if isinstance(pred, IntentPrediction):
                is_correct = pred.predicted_intent == pred.true_intent
                error_type = (
                    None
                    if is_correct
                    else self._classify_error(pred.predicted_intent, pred.true_intent)
                )
                pred.is_correct = is_correct
                pred.error_type = error_type
                validated.append(pred)
            elif isinstance(pred, dict):
                prediction = IntentPrediction(
                    answer_text=pred.get("answer_text", pred.get("answer", "")),
                    predicted_intent=pred.get("predicted_intent", pred.get("predicted", "unknown")),
                    true_intent=pred.get("true_intent", pred.get("true", "unknown")),
                    confidence=pred.get("confidence", 0.5),
                    is_correct=pred.get("predicted_intent", pred.get("predicted"))
                    == pred.get("true_intent", pred.get("true")),
                    error_type=(
                        None
                        if pred.get("predicted_intent", pred.get("predicted"))
                        == pred.get("true_intent", pred.get("true"))
                        else self._classify_error(
                            pred.get("predicted_intent", pred.get("predicted", "unknown")),
                            pred.get("true_intent", pred.get("true", "unknown")),
                        )
                    ),
                )
                validated.append(prediction)
        return validated

    def _calculate_per_intent_accuracy(
        self,
        predictions: List[IntentPrediction],
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate accuracy per intent category."""
        intent_stats: Dict[str, Dict[str, int]] = {}

        for p in predictions:
            if p.true_intent not in intent_stats:
                intent_stats[p.true_intent] = {"correct": 0, "total": 0}
            intent_stats[p.true_intent]["total"] += 1
            if p.is_correct:
                intent_stats[p.true_intent]["correct"] += 1

        return {
            intent: {
                "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
                "correct": stats["correct"],
                "total": stats["total"],
            }
            for intent, stats in intent_stats.items()
        }

    def _build_confusion_matrix(
        self,
        predictions: List[IntentPrediction],
    ) -> Dict[str, Dict[str, int]]:
        """Build confusion matrix from predictions."""
        matrix: Dict[str, Dict[str, int]] = {}

        for p in predictions:
            if p.predicted_intent not in matrix:
                matrix[p.predicted_intent] = {}
            matrix[p.predicted_intent][p.true_intent] = (
                matrix[p.predicted_intent].get(p.true_intent, 0) + 1
            )

        return matrix

    def _identify_common_errors(
        self,
        predictions: List[IntentPrediction],
    ) -> List[Dict[str, Any]]:
        """Identify common error patterns."""
        errors: Dict[str, List[Dict[str, Any]]] = {}

        for p in predictions:
            if not p.is_correct:
                error_key = f"{p.predicted_intent}->{p.true_intent}"
                if error_key not in errors:
                    errors[error_key] = []
                errors[error_key].append(
                    {
                        "answer": p.answer_text[:50],
                        "confidence": p.confidence,
                    }
                )

        # Sort by frequency and return top errors
        sorted_errors = sorted(errors.items(), key=lambda x: len(x[1]), reverse=True)
        return [
            {
                "error_pattern": error_key,
                "count": len(error_list),
                "examples": error_list[:3],
            }
            for error_key, error_list in sorted_errors[:5]
        ]

    def _generate_improvement_suggestions(
        self,
        predictions: List[IntentPrediction],
        per_intent_accuracy: Dict[str, Dict[str, Any]],
        confusion_matrix: Dict[str, Dict[str, int]],
    ) -> List[str]:
        """Generate improvement suggestions based on analysis."""
        suggestions = []

        # Check for low accuracy intents
        low_accuracy_intents = [
            intent for intent, stats in per_intent_accuracy.items() if stats["accuracy"] < 0.8
        ]
        for intent in low_accuracy_intents:
            suggestions.append(
                f"Improve detection for '{intent}' intent "
                f"(accuracy: {per_intent_accuracy[intent]['accuracy']:.1%})"
            )

        # Check for confusion patterns
        for predicted, trues in confusion_matrix.items():
            for true_intent, count in trues.items():
                if predicted != true_intent and count >= 3:
                    suggestions.append(
                        f"Reduce confusion between '{predicted}' and '{true_intent}' "
                        f"({count} occurrences)"
                    )

        # General suggestions
        if not suggestions:
            suggestions.append("Intent detection accuracy is good. Focus on edge cases.")

        return suggestions

    def _classify_error(self, predicted: str, true: str) -> str:
        """Classify the type of error."""
        if predicted == "no_response" and true != "no_response":
            return "missed_response"
        if true == "no_response" and predicted != "no_response":
            return "false_positive"
        if predicted in ("objection", "redirect") and true not in ("objection", "redirect"):
            return "over_classification"
        if true in ("objection", "redirect") and predicted not in ("objection", "redirect"):
            return "under_classification"
        return "misclassification"

    def export_results(self, filepath: str) -> None:
        """Export analysis results to JSON file."""
        report = self.analyze_intents([])  # Get empty report to trigger analysis
        data = {
            "total_predictions": report.total_predictions,
            "accuracy": report.accuracy,
            "per_intent_accuracy": report.per_intent_accuracy,
            "common_errors": report.common_errors,
            "improvement_suggestions": report.improvement_suggestions,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Results exported to %s", filepath)
