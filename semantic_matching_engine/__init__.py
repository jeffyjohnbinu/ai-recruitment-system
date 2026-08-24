"""
Semantic Matching Engine
Day 12 deliverable — Zecpath AI Job Portal

Moves resume <-> job matching beyond keyword overlap (ats_engine/matcher.py)
to embedding-based semantic similarity, compared section-by-section across
Skills, Experience, and Projects.
"""

from .embeddings import get_embedder
from .matcher import SemanticMatchingEngine
from .reports import AccuracyReport, ValidationRow, generate_accuracy_report
from .similarity import compute_similarity
from .thresholds import ThresholdConfig, tune_match_threshold

__all__ = [
    "SemanticMatchingEngine",
    "get_embedder",
    "compute_similarity",
    "ThresholdConfig",
    "tune_match_threshold",
    "generate_accuracy_report",
    "AccuracyReport",
    "ValidationRow",
]
__version__ = "1.0.0"
