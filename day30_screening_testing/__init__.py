"""
Day 30 Screening System Testing & Optimization

Comprehensive testing framework for Zecpath AI screening system.
Validates system performance and improves real-world behavior.

Key capabilities:
- Simulated AI screening calls with realistic conversation flow
- Human judgment comparison using Day 17 fixtures as ground truth
- Intent detection improvements
- Scoring threshold tuning
- False rejection reduction through model optimization
- Generated testing reports and performance metrics
"""

from .conversation_simulation import (
    ConversationSimulator,
    SimulatedCallResult,
    SimulationBatchResult,
)
from .intent_analyzer import (
    IntentAccuracyReport,
    IntentAnalyzer,
    IntentPrediction,
    IntentThresholdConfig,
)
from .model_optimizer import FalseRejectionCase, ModelOptimizationReport, ModelOptimizer
from .scoring_threshold_tuner import ScoringThresholdTuner, ThresholdTestResult, TuningReport
from .test_report_generator import (
    ImprovementAnalysis,
    PerformanceMetrics,
    TestCaseResult,
    TestReport,
    TestReportGenerator,
)

__all__ = [
    # Core classes
    "ConversationSimulator",
    "IntentAnalyzer",
    "ModelOptimizer",
    "ScoringThresholdTuner",
    "TestReportGenerator",
    # Dataclasses
    "SimulatedCallResult",
    "SimulationBatchResult",
    "IntentPrediction",
    "IntentAccuracyReport",
    "IntentThresholdConfig",
    "FalseRejectionCase",
    "ModelOptimizationReport",
    "ThresholdTestResult",
    "TuningReport",
    "TestCaseResult",
    "PerformanceMetrics",
    "ImprovementAnalysis",
    "TestReport",
]
