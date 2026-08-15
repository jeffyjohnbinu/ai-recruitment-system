"""
Resume Text Extraction Engine
Day 5 deliverable — Zecpath AI Job Portal

Converts raw resume files (PDF / DOCX) into clean, normalized,
structured text that downstream AI modules (parsing, scoring,
matching) can consume reliably.
"""

from .extractor import ResumeExtractionEngine

__all__ = ["ResumeExtractionEngine"]
__version__ = "1.0.0"
