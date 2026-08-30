"""
ATS System Testing
Day 17 deliverable — Zecpath AI Job Portal

Validates end-to-end ATS accuracy, reliability, and role adaptability by
running a curated set of resume/job-description fixtures (tech, non-tech,
fresher, senior) through the real pipeline (Days 5-16 engines, when
importable) and comparing the AI output against a manually-reviewed
ground truth. Produces precision/recall/F1 metrics, a mismatch log, and
a prioritized improvement backlog.
"""

from .harness import ATSTestHarness
from .metrics import MetricsReport, compute_metrics

__all__ = ["ATSTestHarness", "MetricsReport", "compute_metrics"]
__version__ = "1.0.0"
