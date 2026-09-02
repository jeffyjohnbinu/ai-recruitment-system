"""
eligibility_decision_engine
============================
Day 21 deliverable — Zecpath pipeline.

Decides which candidates qualify for AI screening calls, based on
ATS output + recruiter-defined per-job rules.

Public API:
    load_ats_results(path)      -> list[CandidateInput]
    load_job_rules(path)        -> RuleConfig
    evaluate_candidate(c, rules)-> CandidateEligibilityResult
    evaluate_batch(cands, rules)-> list[CandidateEligibilityResult]
    build_envelope(results, ...) -> dict   (Day 7 metadata envelope)
"""

from .engine import evaluate_batch, evaluate_candidate
from .envelope import build_envelope
from .loaders import load_ats_results, load_job_rules
from .schema import CandidateEligibilityResult, CandidateInput, EligibilityStatus, RuleConfig

__version__ = "1.0.0"
__day__ = "Day 21"

__all__ = [
    "CandidateInput",
    "RuleConfig",
    "CandidateEligibilityResult",
    "EligibilityStatus",
    "load_ats_results",
    "load_job_rules",
    "evaluate_candidate",
    "evaluate_batch",
    "build_envelope",
    "__version__",
    "__day__",
]
