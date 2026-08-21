"""
Experience Parsing & Relevance Engine
Day 10 deliverable — Zecpath AI Job Portal

Extracts structured work-experience entries (company, title, dates)
from resume text, computes total experience with gap/overlap
detection, and scores role relevance against a target job.
"""

from .engine import ExperienceParsingEngine
from .gaps import ExperienceTimeline, GapPeriod, OverlapPeriod, analyze_timeline
from .parser import ExperienceEntry, ExperienceParser, ExperienceParseResult
from .relevance import (
    ExperienceRelevanceResult,
    ExperienceRelevanceScorer,
    RoleRelevanceScore,
    title_similarity,
)
from .storage import ExperienceRecord, ExperienceRelevanceRecord

__all__ = [
    "ExperienceParsingEngine",
    "ExperienceParser",
    "ExperienceParseResult",
    "ExperienceEntry",
    "analyze_timeline",
    "ExperienceTimeline",
    "GapPeriod",
    "OverlapPeriod",
    "ExperienceRelevanceScorer",
    "ExperienceRelevanceResult",
    "RoleRelevanceScore",
    "title_similarity",
    "ExperienceRecord",
    "ExperienceRelevanceRecord",
]
__version__ = "1.0.0"
