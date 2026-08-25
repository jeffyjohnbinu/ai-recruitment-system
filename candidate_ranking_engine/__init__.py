"""
Candidate Ranking & Shortlisting Engine
Day 14 deliverable — Zecpath AI Job Portal

Consumes Day 12 Semantic Matching Engine output (per candidate/job
similarity records) and produces a sorted, zoned, recruiter-friendly
ranking: shortlist / review / auto-reject, plus top-N candidate lists.
"""

from .ranker import CandidateRankingEngine, RankingConfig

__all__ = ["CandidateRankingEngine", "RankingConfig"]
__version__ = "1.0.0"
