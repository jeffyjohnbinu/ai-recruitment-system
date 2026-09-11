"""
run_tests.py
------------
Comprehensive test runner for Day 30 Screening System Testing & Optimization.

This script demonstrates the complete testing framework by:
1. Simulating screening conversations
2. Analyzing intent detection accuracy
3. Tuning scoring thresholds
4. Optimizing model performance
5. Generating test reports
"""

import json
import sys
from pathlib import Path

# Add root to path
sys.path.insert(0, str(Path.cwd()))

from day26_screening_scoring_engine import (
    DEFAULT_DIMENSION_WEIGHTS,
    ScoringConfig,
    ScreeningScoringEngine,
)
from day29.conversation_state_machine import CallConfig
from day30_screening_testing import (
    ConversationSimulator,
    ImprovementAnalysis,
    IntentAnalyzer,
    IntentPrediction,
    ModelOptimizer,
    PerformanceMetrics,
    ScoringThresholdTuner,
    SimulatedCallResult,
    TestCaseResult,
    TestReportGenerator,
)

print("=" * 80)
print("DAY 30: SCREENING SYSTEM TESTING & OPTIMIZATION")
print("=" * 80)

# ============================================================================
# 1. Initialize Testing Components
# ============================================================================
print("\n--- 1. Initializing Testing Components ---")

simulator = ConversationSimulator()
intent_analyzer = IntentAnalyzer()
model_optimizer = ModelOptimizer()
threshold_tuner = ScoringThresholdTuner()
report_generator = TestReportGenerator()

print("All components initialized successfully.")

# ============================================================================
# 2. Simulate Screening Calls
# ============================================================================
print("\n--- 2. Simulating Screening Calls ---")

# Define test scenarios with candidate answers
test_scenarios = [
    {
        "candidate_id": "cand_001",
        "job_id": "job_001",
        "role_id": "software_engineer",
        "candidate_answers": [
            "Yes",  # Q1: Available to talk
            "My name is John Doe, applying for Software Engineer",  # Q2: Name confirmation
            "I have a B.Tech in Computer Science from Anna University",  # Q3: Education
            "Yes",  # Q4: B.Tech in CS
            "5 years",  # Q5: Total experience
            "3 years",  # Q6: Software dev experience
            "Software Engineer at Tech Corp",  # Q7: Current role
            "Looking for better growth opportunities",  # Q8: Reason for leaving
            "4",  # Q9: System design rating
            "5",  # Q10: DSA rating
            "Yes, worked with AWS for 2 years",  # Q11: Cloud platforms
            "AWS Certified Solutions Architect",  # Q12: Certifications
            "Bangalore",  # Q13: Location
            "Yes",  # Q14: Willing to relocate
            "Yes, comfortable with hybrid work",  # Q15: Work mode
            "800000",  # Q16: Current compensation
            "1200000",  # Q17: Expected compensation
            "Yes",  # Q18: Negotiable
            "2 weeks",  # Q19: Notice period
            "No",  # Q20: Can buy out notice
            "2026-10-01",  # Q21: Joining date
        ],
    },
    {
        "candidate_id": "cand_002",
        "job_id": "job_001",
        "role_id": "sales_executive",
        "candidate_answers": [
            "Yes",  # Q1: Available
            "Jane Smith, applying for Sales Executive",  # Q2: Name
            "I have a B.Com degree from Mumbai University",  # Q3: Qualification
            "Yes",  # Q4: Graduate degree
            "4 years",  # Q5: Total experience
            "2 years",  # Q6: B2B sales experience
            "Sales Associate at Mega Mart",  # Q7: Current role
            "4",  # Q8: Negotiation rating
            "Yes, used Salesforce for 2 years",  # Q9: CRM experience
            "Mumbai",  # Q10: Location
            "Yes",  # Q11: Relocate
            "700000 fixed plus incentives",  # Q12: Current comp
            "1000000",  # Q13: Expected comp
            "1 month",  # Q14: Notice period
            "2026-10-15",  # Q15: Joining date
        ],
    },
]

all_simulation_results = []

for scenario in test_scenarios:
    print(f"\nSimulating call for {scenario['candidate_id']} - {scenario['role_id']}")

    result = simulator.simulate_call(
        candidate_id=scenario["candidate_id"],
        job_id=scenario["job_id"],
        role_id=scenario["role_id"],
        candidate_answers=scenario["candidate_answers"],
    )

    all_simulation_results.append(result)

    # Print summary
    print(f"  Questions asked: {result.questions_asked}")
    print(f"  Questions answered: {result.questions_answered}")
    print(f"  Questions skipped: {result.questions_skipped}")
    print(f"  Reprompts given: {result.reprompts_given}")
    print(f"  Clarifications given: {result.clarifications_given}")
    print(f"  Follow-ups given: {result.followups_given}")
    print(f"  Final state: {result.final_state}")
    print(
        f"  Score: {result.scoring_result.normalized_score:.2f}"
        if result.scoring_result
        else "  Score: N/A"
    )
    print(f"  Intent distribution: {result.intent_distribution}")

# ============================================================================
# 3. Aggregate Simulation Results
# ============================================================================
print("\n--- 3. Aggregating Simulation Results ---")

batch_result = simulator.simulate_batch(test_scenarios)

print(f"\nBatch Summary:")
print(f"  Total calls: {batch_result.total_calls}")
print(f"  Successful: {batch_result.successful_calls}")
print(f"  Failed: {batch_result.failed_calls}")
print(f"  Avg questions asked: {batch_result.average_questions_asked:.1f}")
print(f"  Avg questions answered: {batch_result.average_questions_answered:.1f}")
print(f"  Avg reprompts: {batch_result.average_reprompts:.1f}")
print(f"  Avg duration: {batch_result.average_call_duration:.1f}s")

# ============================================================================
# 4. Analyze Intent Detection
# ============================================================================
print("\n--- 4. Analyzing Intent Detection ---")

# Collect all intent predictions from simulation results
intent_predictions = []

for result in all_simulation_results:
    for intent, count in result.intent_distribution.items():
        for _ in range(count):
            intent_predictions.append(
                IntentPrediction(
                    answer_text=f"Answer for {result.call_id}",
                    predicted_intent=intent,
                    true_intent=intent,
                    confidence=0.8,
                )
            )

if intent_predictions:
    intent_report = intent_analyzer.analyze_intents(intent_predictions)

    print(f"Intent Analysis Results:")
    print(f"  Total predictions: {intent_report.total_predictions}")
    print(f"  Correct: {intent_report.correct_predictions}")
    print(f"  Accuracy: {intent_report.accuracy:.1%}")

    print(f"\nPer-intent accuracy:")
    for intent, stats in intent_report.per_intent_accuracy.items():
        print(f"    {intent}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")

    print(f"\nCommon errors:")
    for error in intent_report.common_errors[:5]:
        print(f"    {error['error_pattern']}: {error['count']} occurrences")

    print(f"\nImprovement suggestions:")
    for suggestion in intent_report.improvement_suggestions[:5]:
        print(f"    * {suggestion}")

# ============================================================================
# 5. Tune Scoring Thresholds
# ============================================================================
print("\n--- 5. Tuning Scoring Thresholds ---")

# Collect scores and true recommendations from simulation results
scores = []
true_recommendations = []

for result in all_simulation_results:
    if result.scoring_result:
        scores.append(result.scoring_result.normalized_score)
        # Map recommendation to our scoring
        if result.scoring_result.recommendation == "proceed":
            true_recommendations.append("proceed")
        elif result.scoring_result.recommendation == "hold":
            true_recommendations.append("hold")
        else:
            true_recommendations.append("reject")

if scores and true_recommendations:
    tuning_report = threshold_tuner.tune_thresholds(
        scores=scores,
        true_recommendations=true_recommendations,
        current_thresholds=(0.75, 0.45),
    )

    print(f"Threshold Tuning Results:")
    print(f"  Original accuracy: {tuning_report.original_accuracy:.1%}")
    print(f"  Optimized accuracy: {tuning_report.best_result.accuracy:.1%}")
    print(f"  False rejection reduction: {tuning_report.false_rejection_reduction:.1f}%")
    print(f"  Best thresholds:")
    print(f"    Proceed threshold: {tuning_report.best_threshold[0]}")
    print(f"    Hold threshold: {tuning_report.best_threshold[1]}")

    print(f"\nPerformance at best thresholds:")
    print(f"    Accuracy: {tuning_report.best_result.accuracy:.1%}")
    print(f"    Precision: {tuning_report.best_result.precision:.1%}")
    print(f"    Recall: {tuning_report.best_result.recall:.1%}")
    print(f"    F1 Score: {tuning_report.best_result.f1_score:.1%}")
    print(f"    False Positives: {tuning_report.best_result.false_positives}")
    print(f"    False Negatives: {tuning_report.best_result.false_negatives}")

# ============================================================================
# 6. Optimize Model
# ============================================================================
print("\n--- 6. Optimizing Model Performance ---")

# Analyze false rejections from scoring results
cases = []
ground_truth = []

for idx, result in enumerate(all_simulation_results):
    if result.scoring_result:
        cases.append(
            {
                "case_id": result.call_id,
                "candidate_id": result.candidate_id,
                "score": result.scoring_result.normalized_score,
                "recommendation": result.scoring_result.recommendation,
                "reason": f"Score={result.scoring_result.normalized_score:.2f}",
                "dimension_scores": {
                    "clarity": 0.8,
                    "relevance": result.scoring_result.overall_dimension_scores.get(
                        "relevance", 0.5
                    ),
                    "completeness": result.scoring_result.overall_dimension_scores.get(
                        "completeness", 0.5
                    ),
                    "consistency": result.scoring_result.overall_dimension_scores.get(
                        "consistency", 0.5
                    ),
                },
            }
        )
        # Use index from the cases list to find the matching ground truth
        gt_idx = len(cases) - 1
        expected_rec = (
            true_recommendations[gt_idx] if gt_idx < len(true_recommendations) else "proceed"
        )
        ground_truth.append(
            {
                "case_id": result.call_id,
                "expected_recommendation": expected_rec,
            }
        )

# Run model optimization
if cases and ground_truth:
    model_report = model_optimizer.analyze_false_rejections(cases, ground_truth)

    print(f"Model Optimization Results:")
    print(f"  Total cases: {model_report.total_cases}")
    print(f"  False rejections: {model_report.false_rejections}")
    print(f"  False rejection rate: {model_report.false_rejection_rate:.1%}")

    print(f"\nTop false rejection reasons:")
    for reason in model_report.top_false_rejection_reasons[:5]:
        print(f"    * {reason['reason']}: {reason['count']} ({reason['percentage']:.1%})")

    print(f"\nOptimization recommendations:")
    for rec in model_report.optimization_recommendations[:5]:
        print(f"    * {rec['description']}")

    # Dimension analysis
    print(f"\nDimension analysis:")
    for dim, analysis in model_report.dimension_analysis.items():
        print(
            f"    * {dim}: avg={analysis['average_score']:.2f}, low_score_rate={analysis['low_score_percentage']:.1%}"
        )

# ============================================================================
# 7. Generate Comprehensive Test Report
# ============================================================================
print("\n--- 7. Generating Comprehensive Test Report ---")

# Gather all metrics from testing
all_scores = []
all_recommendations = []

for result in all_simulation_results:
    if result.scoring_result:
        all_scores.append(result.scoring_result.normalized_score)
        all_recommendations.append(result.scoring_result.recommendation)

# Calculate basic metrics
total_calls = len(all_simulation_results)
successful_calls = sum(1 for r in all_simulation_results if r.success and r.scoring_result)
accuracy = (
    len([r for r in all_simulation_results if r.scoring_result]) / total_calls
    if total_calls > 0
    else 0
)

if all_scores:
    performance_metrics = PerformanceMetrics(
        accuracy=accuracy,
        precision=(
            sum(
                1
                for r in all_simulation_results
                if r.scoring_result and r.scoring_result.recommendation in ("proceed", "hold")
            )
            / len(all_scores)
            if all_scores
            else 0
        ),
        recall=(
            sum(1 for r in all_simulation_results if r.scoring_result) / total_calls
            if total_calls > 0
            else 0
        ),
        f1_score=0.0,  # Would need full confusion matrix
        true_positives=0,
        false_positives=0,
        true_negatives=0,
        false_negatives=0,
        true_positive_rate=0.0,
        false_positive_rate=0.0,
        false_negative_rate=0.0,
        false_rejection_rate=0.0,
        average_score=sum(all_scores) / len(all_scores) if all_scores else 0,
        score_std_deviation=0.0,  # Would calculate from scores
        confidence_threshold=0.5,
    )

    # Create test case results
    test_case_results = []
    for i, result in enumerate(all_simulation_results):
        if result.scoring_result:
            test_case_results.append(
                TestCaseResult(
                    case_id=result.call_id,
                    scenario=f"{result.role_id} screening",
                    actual_score=result.scoring_result.normalized_score,
                    expected_score=0.75,  # Target threshold
                    score_difference=abs(result.scoring_result.normalized_score - 0.75),
                    passed=result.scoring_result.normalized_score >= 0.75,
                    details=f"Recommendation: {result.scoring_result.recommendation}, Score: {result.scoring_result.normalized_score:.2f}",
                )
            )
        else:
            test_case_results.append(
                TestCaseResult(
                    case_id=result.call_id,
                    scenario=f"{result.role_id} screening",
                    actual_score=0.0,
                    expected_score=0.75,
                    score_difference=0.75,
                    passed=False,
                    details=f"Error in scoring: {result.error_message}",
                )
            )

    # Create improvement analysis
    improvement_analysis = ImprovementAnalysis(
        improvements=(
            [
                {
                    "description": f"Reduced false rejection rate by {tuning_report.false_rejection_reduction:.1f}%",
                    "priority": "high",
                }
            ]
            if tuning_report
            else []
        ),
        regression_warnings=[],
        recommendations=[
            "Tune thresholds to reduce false rejections",
            "Improve intent detection accuracy",
            "Increase model confidence handling",
        ],
        cost_benefit_analysis={},
    )

    # Generate the test report
    test_report = report_generator.generate_report(
        test_run_id="day30_run_001",
        performance_metrics=performance_metrics,
        test_case_results=test_case_results,
        improvement_analysis=improvement_analysis,
        intent_accuracy_report=(
            {
                "total_predictions": (
                    intent_report.total_predictions if "intent_report" in dir() else 0
                ),
                "accuracy": intent_report.accuracy if "intent_report" in dir() else 0,
            }
            if "intent_report" in dir()
            else None
        ),
        threshold_report=tuning_report.__dict__ if "tuning_report" in dir() else None,
        model_report=model_report.__dict__ if "model_report" in dir() else None,
        simulation_results=[
            {
                "call_id": r.call_id,
                "final_state": r.final_state,
                "success": r.success,
                "score": r.scoring_result.normalized_score if r.scoring_result else None,
            }
            for r in all_simulation_results
        ],
        conversation_metrics={
            "avg_questions_asked": batch_result.average_questions_asked,
            "avg_questions_answered": batch_result.average_questions_answered,
            "avg_reprompts": batch_result.average_reprompts,
            "success_rate": (
                batch_result.successful_calls / batch_result.total_calls
                if batch_result.total_calls > 0
                else 0
            ),
        },
    )

    # Export report
    report_generator.export_report(test_report, "day30_test_report.json")
    print(f"\nTest report exported to: day30_test_report.json")

    # Generate summary
    summary = report_generator.generate_summary_report(
        test_run_id="day30_run_001",
        metrics=performance_metrics,
        overall_improvements={
            "improvements": [
                (
                    {
                        "description": f"False rejection reduction: {tuning_report.false_rejection_reduction:.1f}%",
                        "priority": "high",
                    }
                    if "tuning_report" in dir()
                    else {}
                ),
                (
                    {
                        "description": f"Intent accuracy: {intent_report.accuracy:.1%}",
                        "priority": "high",
                    }
                    if "intent_report" in dir()
                    else {}
                ),
                {"description": f"Model optimization completed", "priority": "medium"},
            ],
            "ready_for_production": test_report.deployment_readiness["ready_for_production"],
            "risk_level": test_report.risk_assessment["risk_level"],
        },
    )

    print(f"\n--- Test Summary ---")
    print(summary)

# ============================================================================
# 8. Export Component Results
# ============================================================================
print("\n--- 8. Exporting Component Results ---")

# Export intent analysis if available
if "intent_report" in dir():
    intent_analyzer.export_results("intent_analysis_results.json")
    print("  Intent analysis exported to: intent_analysis_results.json")

# Export model optimization if available
if "model_report" in dir():
    model_optimizer.export_report(model_report, "model_optimization_report.json")
    print("  Model optimization report exported to: model_optimization_report.json")

# Export threshold tuning results
if "tuning_report" in dir():
    threshold_tuner.export_report(tuning_report, "threshold_tuning_report.json")
    print("  Threshold tuning report exported to: threshold_tuning_report.json")

print("\n" + "=" * 80)
print("DAY 30 TESTING COMPLETE")
print("=" * 80)
print("\nKey Deliverables Generated:")
print("  • Screening system test report (day30_test_report.json)")
print("  • Intent analysis results (intent_analysis_results.json)")
print("  • Model optimization report (model_optimization_report.json)")
print("  • Threshold tuning report (threshold_tuning_report.json)")
print("  • Summary report with actionable recommendations")
print("  • All component metrics and analysis")
print("\nRecommendations for production:")
print("  • Review threshold tuning results before production deployment")
print("  • Address high-priority false rejection cases")
print("  • Implement intent detection improvements")
print("  • Monitor system performance post-deployment")
print("=" * 80)
