# Day 30 – Screening System Testing & Optimization

Comprehensive testing framework for Zecpath AI Screening System.
Validates real-world performance, analyzes candidate intent detection, tunes recommendation scoring thresholds, optimizes AI model parameters to minimize false rejections, and generates executive test reports.

---

## 📁 Directory Layout

```
day30_screening_testing/
├── README.md                      # Module documentation
├── __init__.py                    # Module exports and package initialization
├── conversation_simulation.py     # Multi-turn AI screening call simulator
├── intent_analyzer.py             # Candidate intent classification analyzer & accuracy tracker
├── model_optimizer.py             # Model optimization engine for false rejection reduction
├── scoring_threshold_tuner.py     # Grid-search threshold tuner for recommendation scoring
├── test_report_generator.py      # Comprehensive test report generator & risk assessor
└── run_tests.py                   # Master test runner & demonstration script
```

---

## ⚙️ Core Architecture & Components

### 1. `ConversationSimulator` (`conversation_simulation.py`)
- Integrates **Day 29 `ConversationStateMachine`** with **Day 26 `ScreeningScoringEngine`**.
- Simulates realistic multi-turn conversations for roles like `software_engineer` and `sales_executive`.
- Tracks conversation metrics: questions asked/answered/skipped, reprompts given, clarifications, follow-ups, and intent distributions.

### 2. `IntentAnalyzer` (`intent_analyzer.py`)
- Evaluates candidate response classification against ground truth labels across key intent categories (`boolean`, `numeric`, `answer`, `no_response`, `clarification_request`, `objection`, `redirect`).
- Generates confusion matrices, identifies common error patterns, and produces actionable recommendations for intent model tuning.

### 3. `ScoringThresholdTuner` (`scoring_threshold_tuner.py`)
- Conducts grid-search optimization across candidate `proceed` and `hold` score thresholds.
- Evaluates precision, recall, F1 score, false positive rate, and false rejection rate.
- Identifies optimal thresholds to maximize candidate evaluation accuracy while strictly minimizing false rejections.

### 4. `ModelOptimizer` (`model_optimizer.py`)
- Analyzes candidate cases incorrectly rejected by the screening engine.
- Evaluates dimension-wise score distributions (`clarity`, `relevance`, `completeness`, `consistency`).
- Computes optimized dimension weights and generates exception-handling rules for human recruiter escalation on edge cases.

### 5. `TestReportGenerator` (`test_report_generator.py`)
- Aggregates performance metrics, simulation results, intent analysis, threshold tuning, and model optimization into unified JSON reports.
- Computes production deployment readiness scores, risk assessments, and executive summaries.

---

## 🚀 Quick Start & Usage

### Running the Test Runner
To execute the full simulation, threshold tuning, model optimization, and report generation pipeline:

```bash
python day30_screening_testing/run_tests.py
```

### Generated Deliverables & Output Files
Running `run_tests.py` produces the following output reports in your working directory:
- **`day30_test_report.json`**: Complete system test report with deployment readiness score.
- **`intent_analysis_results.json`**: Intent classification accuracy and error analysis.
- **`model_optimization_report.json`**: False rejection analysis and dimension weight optimization.
- **`threshold_tuning_report.json`**: Grid-search tuning metrics and recommended threshold boundaries.

### Running Pytest Unit Tests
To execute all automated unit tests for Day 30 and the recruitment system:

```bash
pytest
```

---

## 📊 Performance & Optimization Summary

| Metric | Initial Baseline | Post-Optimization Target |
| :--- | :---: | :---: |
| **Intent Detection Accuracy** | 85.0% | **100.0%** |
| **Batch Simulation Success Rate** | 50.0% | **100.0%** |
| **False Rejection Rate** | Variable | **0.0%** |
| **Recommended Proceed Threshold** | `>= 0.75` | **`>= 0.55`** |
| **Recommended Hold Threshold** | `>= 0.45` | **`>= 0.30`** |

---

## 🛠 Integration Example

```python
from day30_screening_testing import (
    ConversationSimulator,
    IntentAnalyzer,
    ModelOptimizer,
    ScoringThresholdTuner,
    TestReportGenerator,
)

# 1. Simulate screening call
simulator = ConversationSimulator()
result = simulator.simulate_call(
    candidate_id="cand_001",
    job_id="job_001",
    role_id="software_engineer",
    candidate_answers=["Yes", "John Doe", "B.Tech CS", "Yes", "5 years", ...],
)

# 2. Tune scoring thresholds
tuner = ScoringThresholdTuner()
report = tuner.tune_thresholds(
    scores=[result.scoring_result.normalized_score],
    true_recommendations=["proceed"],
)
print("Best thresholds:", report.best_threshold)
```
