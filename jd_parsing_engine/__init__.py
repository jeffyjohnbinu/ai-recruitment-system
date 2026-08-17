"""
Job Description Parsing System
Day 6 deliverable — Zecpath AI Job Portal

Converts raw employer job description text into clean, normalized,
structured "job requirement objects" that downstream AI modules
(ATS matching, screening, interview question generation) can consume
reliably — mirroring the shape of resume_extraction_engine's output
so the two sides of the pipeline (resume <-> JD) speak the same
structured language.
"""

from .parser import JDParsingEngine

__all__ = ["JDParsingEngine"]
__version__ = "1.0.0"
