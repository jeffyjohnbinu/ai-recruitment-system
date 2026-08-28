"""
Fairness, Normalization & Bias Reduction Engine
Day 15 deliverable — Zecpath AI Job Portal

Improves fairness and standardizes resume evaluation by:
  - normalizing heterogeneous candidate profiles to a canonical schema
  - masking non-essential personal attributes before scoring
  - reducing over-dependence on raw keyword matching
  - normalizing final scores across a candidate pool
  - auditing bias indicators (four-fifths rule) across supplied groups
"""

from .bias_auditor import BiasAuditor, BiasAuditReport, GroupStat
from .engine import CandidateFairnessResult, FairnessBiasEngine, FairnessProcessingResult
from .keyword_balancer import KeywordBalanceReport, KeywordDependencyReducer
from .masker import AttributeMasker, MaskingResult
from .normalizer import CanonicalCandidateProfile, ResumeNormalizer
from .score_normalizer import NormalizedScoreRecord, ScoreNormalizer
from .storage import FairnessProcessingRecord, FairnessResultStore

__all__ = [
    "BiasAuditor",
    "BiasAuditReport",
    "GroupStat",
    "CandidateFairnessResult",
    "FairnessBiasEngine",
    "FairnessProcessingResult",
    "KeywordBalanceReport",
    "KeywordDependencyReducer",
    "AttributeMasker",
    "MaskingResult",
    "CanonicalCandidateProfile",
    "ResumeNormalizer",
    "NormalizedScoreRecord",
    "ScoreNormalizer",
    "FairnessProcessingRecord",
    "FairnessResultStore",
]

__version__ = "1.0.0"
