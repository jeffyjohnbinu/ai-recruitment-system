"""
test_report_generator.py
-----------------------
Generates comprehensive testing reports for the screening system.

Combines outputs from all testing components into a unified
display suitable for stakeholders and decision-makers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger("day30_screening_testing.test_report_generator")


@dataclass
class TestCaseResult:
    """Individual test case result."""

    __test__ = False

    case_id: str
    scenario: str
    actual_score: float
    expected_score: float
    score_difference: float
    passed: bool
    details: str


@dataclass
class PerformanceMetrics:
    """Key performance metrics."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    true_positive_rate: float
    false_positive_rate: float
    false_negative_rate: float
    false_rejection_rate: float
    average_score: float
    score_std_deviation: float
    confidence_threshold: float


@dataclass
class ImprovementAnalysis:
    """Analysis of improvements achieved."""

    improvements: List[Dict[str, Any]]
    regression_warnings: List[str]
    recommendations: List[str]
    cost_benefit_analysis: Dict[str, Any]


@dataclass
class TestReport:
    """Complete test report."""

    test_run_id: str
    test_run_date: str
    system_under_test: str
    test_suite: str
    performance_metrics: PerformanceMetrics
    test_case_results: List[TestCaseResult]
    improvement_analysis: ImprovementAnalysis
    recommendations: List[str]
    risk_assessment: Dict[str, Any]
    deployment_readiness: Dict[str, Any]
    executive_summary: Dict[str, Any]


class TestReportGenerator:
    __test__ = False

    """
    Generates comprehensive testing reports for the screening system.

    Combines outputs from all testing components into a unified
    display suitable for stakeholders and decision-makers.

    Usage:
        generator = TestReportGenerator()
        report = generator.generate_report(
            test_run_id="run_001",
            performance_metrics=metrics,
            test_results=results,
            improvement_analysis=improvements,
            intent_accuracy=intent_report,
            threshold_report=threshold_report,
            model_report=model_report,
        )
        generator.export_report(report, "report.json")
    """

    def __init__(self) -> None:
        """Initialize report generator."""
        pass

    def generate_report(
        self,
        test_run_id: str,
        performance_metrics: PerformanceMetrics,
        test_case_results: List[TestCaseResult],
        improvement_analysis: ImprovementAnalysis,
        intent_accuracy_report: Dict[str, Any] = None,
        threshold_report: Dict[str, Any] = None,
        model_report: Dict[str, Any] = None,
        simulation_results: List[Dict[str, Any]] = None,
        conversation_metrics: Dict[str, Any] = None,
    ) -> TestReport:
        """
        Generate comprehensive test report.

        Args:
            test_run_id: Unique identifier for this test run.
            performance_metrics: Overall system performance metrics.
            test_case_results: Results of individual test cases.
            improvement_analysis: Analysis of improvements.
            intent_accuracy_report: Intent detection accuracy report.
            threshold_report: Threshold tuning report.
            model_report: Model optimization report.
            simulation_results: Results from conversation simulations.
            conversation_metrics: Metrics from conversation flows.

        Returns:
            Complete test report.
        """
        logger.info("Generating comprehensive test report: %s", test_run_id)

        # Calculate key metrics
        passed_cases = sum(1 for r in test_case_results if r.passed)
        total_cases = len(test_case_results)
        overall_accuracy = passed_cases / total_cases if total_cases > 0 else 0.0

        # Calculate deployment readiness
        deployment_readiness = self._calculate_deployment_readiness(
            performance_metrics, overall_accuracy, improvement_analysis
        )

        # Calculate risk assessment
        risk_assessment = self._calculate_risk_assessment(
            performance_metrics, improvement_analysis, test_case_results
        )

        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            test_run_id, performance_metrics, improvement_analysis, deployment_readiness
        )

        # Generate overall recommendations
        recommendations = self._generate_recommendations(
            performance_metrics, improvement_analysis, risk_assessment
        )

        report = TestReport(
            test_run_id=test_run_id,
            test_run_date=datetime.now(timezone.utc).isoformat(),
            system_under_test="Zecpath AI Screening System",
            test_suite="Day 30 Comprehensive Testing",
            performance_metrics=performance_metrics,
            test_case_results=test_case_results,
            improvement_analysis=improvement_analysis,
            recommendations=recommendations,
            risk_assessment=risk_assessment,
            deployment_readiness=deployment_readiness,
            executive_summary=executive_summary,
        )

        logger.info("Test report generated: %s", test_run_id)
        return report

    def _calculate_deployment_readiness(
        self,
        metrics: PerformanceMetrics,
        accuracy: float,
        improvement: ImprovementAnalysis,
    ) -> Dict[str, Any]:
        """Calculate deployment readiness score and criteria."""
        readiness_criteria = {
            "accuracy": {
                "target": 0.95,
                "current": accuracy,
                "met": accuracy >= 0.95,
            },
            "false_rejection_rate": {
                "target": 0.1,
                "current": metrics.false_negative_rate,
                "met": metrics.false_negative_rate <= 0.1,
            },
            "precision": {
                "target": 0.9,
                "current": metrics.precision,
                "met": metrics.precision >= 0.9,
            },
            "f1_score": {
                "target": 0.9,
                "current": metrics.f1_score,
                "met": metrics.f1_score >= 0.9,
            },
        }

        # Calculate overall readiness score
        criteria_values = [
            readiness_criteria["accuracy"]["met"],
            readiness_criteria["false_rejection_rate"]["met"],
            readiness_criteria["precision"]["met"],
            readiness_criteria["f1_score"]["met"],
        ]
        overall_score = sum(criteria_values) / len(criteria_values)

        return {
            "ready_for_production": overall_score >= 0.75,
            "readiness_score": overall_score,
            "criteria_met": sum(criteria_values),
            "total_criteria": len(criteria_values),
            "criteria_details": readiness_criteria,
            "blocking_issues": [
                criterion for criterion, details in readiness_criteria.items() if not details["met"]
            ],
        }

    def _calculate_risk_assessment(
        self,
        metrics: PerformanceMetrics,
        improvement: ImprovementAnalysis,
        test_cases: List[TestCaseResult],
    ) -> Dict[str, Any]:
        """Assess risks and potential issues."""
        risks = []
        risk_level = "LOW"

        # Accuracy risk
        if metrics.accuracy < 0.9:
            risks.append(
                {
                    "type": "accuracy",
                    "severity": "HIGH" if metrics.accuracy < 0.8 else "MEDIUM",
                    "description": f"Accuracy below target: {metrics.accuracy:.1%}",
                    "impact": "Potential false positives/negatives in candidate screening",
                }
            )
            if risk_level == "LOW":
                risk_level = "MEDIUM"

        # False rejection risk
        if metrics.false_negative_rate > 0.15:
            risks.append(
                {
                    "type": "false_rejection",
                    "severity": "CRITICAL",
                    "description": f"False rejection rate above safe threshold: {metrics.false_negative_rate:.1%}",
                    "impact": "Good candidates incorrectly rejected",
                }
            )
            risk_level = "HIGH"

        # Precision risk
        if metrics.precision < 0.8:
            risks.append(
                {
                    "type": "precision",
                    "severity": "MEDIUM",
                    "description": f"Precision below acceptable level: {metrics.precision:.1%}",
                    "impact": "Candidates marked as suitable may not be qualified",
                }
            )

        # High false positive risk
        if metrics.false_positive_rate > 0.2:
            risks.append(
                {
                    "type": "false_positive",
                    "severity": "MEDIUM",
                    "description": f"False positive rate above acceptable: {metrics.false_positive_rate:.1%}",
                    "impact": "Qualified candidates incorrectly flagged",
                }
            )

        # Confidence analysis risk
        if metrics.confidence_threshold > 0.5:
            risks.append(
                {
                    "type": "confidence_threshold",
                    "severity": "LOW",
                    "description": f"High confidence threshold may cause over-filtering",
                    "impact": "Potential reduction in candidate diversity",
                }
            )

        return {
            "risk_level": risk_level,
            "total_risks": len(risks),
            "critical_risks": [r for r in risks if r["severity"] == "CRITICAL"],
            "high_risks": [r for r in risks if r["severity"] == "HIGH"],
            "medium_risks": [r for r in risks if r["severity"] == "MEDIUM"],
            "low_risks": [r for r in risks if r["severity"] == "LOW"],
            "risk_distribution": {
                "critical": len([r for r in risks if r["severity"] == "CRITICAL"]),
                "high": len([r for r in risks if r["severity"] == "HIGH"]),
                "medium": len([r for r in risks if r["severity"] == "MEDIUM"]),
                "low": len([r for r in risks if r["severity"] == "LOW"]),
            },
        }

    def _generate_executive_summary(
        self,
        test_run_id: str,
        metrics: PerformanceMetrics,
        improvement: ImprovementAnalysis,
        deployment: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate executive-friendly summary."""
        key_improvements = [
            imp for imp in improvement.improvements if imp.get("priority") in ["high", "critical"]
        ]

        return {
            "test_run": test_run_id,
            "test_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "system_tested": "AI Screening Pipeline",
            "key_findings": [
                f"Accuracy: {metrics.accuracy:.1%}",
                f"False Rejection Rate: {metrics.false_negative_rate:.1%}",
                f"Precision: {metrics.precision:.1%}",
                f"F1 Score: {metrics.f1_score:.1%}",
                f"Overall Score: {metrics.f1_score:.1%}",
            ],
            "critical_improvements": [imp["description"] for imp in key_improvements[:3]],
            "deployment_readiness": {
                "overall_score": deployment["readiness_score"],
                "ready": deployment["ready_for_production"],
                "criteria_met": f"{deployment['criteria_met']}/{deployment['total_criteria']}",
            },
            "risk_level": "LOW" if deployment["ready_for_production"] else "MEDIUM",
            "next_steps": [
                "Review and address blocking issues",
                "Implement highest-priority improvements",
                "Monitor system after deployment",
                "Plan rollback procedures if needed",
            ],
        }

    def _generate_recommendations(
        self,
        metrics: PerformanceMetrics,
        improvement: ImprovementAnalysis,
        risk: Dict[str, Any],
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # High priority recommendations
        if metrics.false_negative_rate > 0.15:
            recommendations.append(
                "IMMEDIATE ACTION: Reduce false rejection rate through threshold tuning and model improvement"
            )

        if metrics.accuracy < 0.9:
            recommendations.append(
                "REVIEW: Improve overall accuracy through better intent detection and scoring models"
            )

        # Medium priority recommendations
        if metrics.precision < 0.8:
            recommendations.append(
                "OPTIMIZE: Adjust precision/recall balance for better candidate matching"
            )

        # Low priority recommendations
        if metrics.confidence_threshold > 0.5:
            recommendations.append(
                "TUNE: Lower confidence threshold to capture more qualified candidates"
            )

        # Add improvement analysis recommendations
        for imp in improvement.improvements:
            if imp.get("priority") == "high":
                recommendations.append(imp["description"])

        # Add risk-based recommendations
        for risk in risk["high_risks"] + risk["critical_risks"]:
            recommendations.append(f"MITIGATE RISK: {risk['description']}")

        if not recommendations:
            recommendations.append("MAINTAIN CURRENT PERFORMANCE: System meeting targets")

        return recommendations

    def export_report(self, report: TestReport, filepath: str) -> None:
        """Export test report to JSON file."""
        data = {
            "test_run_id": report.test_run_id,
            "test_run_date": report.test_run_date,
            "system_under_test": report.system_under_test,
            "test_suite": report.test_suite,
            "performance_metrics": report.performance_metrics.__dict__,
            "test_case_results": [
                {
                    **r.__dict__,
                    "passed": r.passed,
                }
                for r in report.test_case_results
            ],
            "improvement_analysis": {
                **report.improvement_analysis.__dict__,
                "improvements": report.improvement_analysis.improvements,
                "regression_warnings": report.improvement_analysis.regression_warnings,
                "recommendations": report.improvement_analysis.recommendations,
                "cost_benefit_analysis": report.improvement_analysis.cost_benefit_analysis,
            },
            "recommendations": report.recommendations,
            "risk_assessment": report.risk_assessment,
            "deployment_readiness": report.deployment_readiness,
            "executive_summary": report.executive_summary,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Test report exported to %s", filepath)

    def generate_summary_report(
        self,
        test_run_id: str,
        metrics: PerformanceMetrics,
        overall_improvements: Dict[str, Any],
    ) -> str:
        """
        Generate a human-readable summary report.

        Args:
            test_run_id: Test run identifier.
            metrics: Performance metrics.
            overall_improvements: Summary of improvements.

        Returns:
            Human-readable summary text.
        """
        summary_lines = [
            "=" * 80,
            "ZECPATH SCREENING SYSTEM - DAY 30 TEST RESULTS",
            "=" * 80,
            f"Test Run ID: {test_run_id}",
            f"Test Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "PERFORMANCE METRICS:",
            f"  Accuracy: {metrics.accuracy:.1%}",
            f"  Precision: {metrics.precision:.1%}",
            f"  Recall: {metrics.recall:.1%}",
            f"  F1 Score: {metrics.f1_score:.1%}",
            f"  True Positives: {metrics.true_positives}",
            f"  False Positives: {metrics.false_positives}",
            f"  True Negatives: {metrics.true_negatives}",
            f"  False Negatives: {metrics.false_negatives}",
            f"  False Rejection Rate: {metrics.false_negative_rate:.1%}",
            f"  False Positive Rate: {metrics.false_positive_rate:.1%}",
            f"  Average Score: {metrics.average_score:.3f}",
            f"  Score Standard Deviation: {metrics.score_std_deviation:.3f}",
            "",
            "OVERALL IMPROVEMENTS:",
        ]

        for improvement in overall_improvements.get("improvements", []):
            summary_lines.append(f"  * {improvement.get('description', 'N/A')}")

        summary_lines.extend(
            [
                "",
                "DEPLOYMENT READINESS:",
                f"  Ready for Production: {'YES' if overall_improvements.get('ready_for_production', False) else 'NO'}",
                f"  Readiness Score: {overall_improvements.get('readiness_score', 0):.1%}",
                "",
                "RISK ASSESSMENT:",
                f"  Risk Level: {overall_improvements.get('risk_level', 'UNKNOWN')}",
                f"  Critical Issues: {len(overall_improvements.get('critical_risks', []))}",
                f"  High Priority Issues: {len(overall_improvements.get('high_priority_issues', []))}",
                "",
                "KEY RECOMMENDATIONS:",
            ]
        )

        for recommendation in overall_improvements.get("recommendations", [])[:5]:
            summary_lines.append(f"  • {recommendation}")

        summary_lines.extend(
            [
                "",
                "=" * 80,
                "End of Report",
                "=" * 80,
            ]
        )

        return "\n".join(summary_lines)
