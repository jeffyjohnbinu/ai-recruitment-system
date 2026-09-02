"""
Data structures for Eligibility Decision Engine.

Design intent: plain dataclasses, JSON-serializable, alias-tolerant
at load time (see loaders.py). No hidden magic in these classes —
they are the contract the rest of the pipeline (and recruiters'
config files) build against.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional


class EligibilityStatus(str, Enum):
    ELIGIBLE = "eligible"
    REVIEW = "review"
    REJECTED = "rejected"


@dataclass
class CandidateInput:
    """Normalized candidate record, as produced by ATS output loader."""

    candidate_id: str
    job_role: str
    ats_score: float  # 0-100
    skills: list[str] = field(default_factory=list)
    experience_years: float = 0.0
    location: Optional[str] = None
    remote_ok: bool = False
    availability: Optional[str] = None  # e.g. "immediate", "2_weeks"
    raw: dict = field(default_factory=dict)  # original record, untouched

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw", None)
        return d


@dataclass
class RuleConfig:
    """Configurable cutoff rules for one job role."""

    job_role: str
    min_ats_score: float = 0.0
    review_band: float = 0.0  # score points below min_ats_score -> REVIEW instead of REJECTED
    mandatory_skills: list[str] = field(default_factory=list)
    min_experience_years: float = 0.0
    max_experience_years: Optional[float] = None
    allowed_locations: list[str] = field(default_factory=list)  # empty = no location restriction
    allow_remote: bool = True
    required_availability: Optional[str] = None  # None = no constraint

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RuleReason:
    """One rule check outcome, for auditability."""

    rule: str
    passed: bool
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CandidateEligibilityResult:
    candidate_id: str
    job_role: str
    status: EligibilityStatus
    ats_score: float
    reasons: list[RuleReason] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "job_role": self.job_role,
            "status": self.status.value,
            "ats_score": self.ats_score,
            "reasons": [r.to_dict() for r in self.reasons],
        }
