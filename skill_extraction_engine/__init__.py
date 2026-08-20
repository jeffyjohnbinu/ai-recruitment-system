"""
Skill Extraction Engine
Day 9 deliverable — Zecpath AI Job Portal

Extracts, normalizes, and confidence-scores technical, business, and
creative skills from cleaned resume text.
"""

from .extractor import SkillExtractionEngine
from .normalizer import SkillRecord
from .storage import SkillExtractionResult

__all__ = ["SkillExtractionEngine", "SkillRecord", "SkillExtractionResult"]
__version__ = "1.0.0"
