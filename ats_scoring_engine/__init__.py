"""
ATS Scoring Formula & Engine
Day 13 deliverable — Zecpath AI Job Portal

Combines Day 9 (skill match), Day 10 (experience relevance), Day 11
(education alignment), and Day 12 (semantic similarity) outputs into a
single, transparent, explainable candidate score using a configurable,
per-role weight system.
"""

from .explainability import build_component_explanations, build_narrative
from .generator import CandidateScoreGenerator, CandidateScoreRequest, load_manifest
from .scoring_engine import ATSScoringEngine, ComponentBreakdown, ScoringResult
from .storage import ResultStore, ScoreRecord, build_score_record
from .weights import COMPONENT_NAMES, WeightProfile, WeightProfileRegistry

__all__ = [
    "ATSScoringEngine",
    "ScoringResult",
    "ComponentBreakdown",
    "WeightProfile",
    "WeightProfileRegistry",
    "COMPONENT_NAMES",
    "ScoreRecord",
    "ResultStore",
    "build_score_record",
    "build_narrative",
    "build_component_explanations",
    "CandidateScoreGenerator",
    "CandidateScoreRequest",
    "load_manifest",
]
__version__ = "1.0.0"
