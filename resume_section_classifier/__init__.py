"""
Resume Section Segmentation / Classification Engine
Day 8 deliverable — Zecpath AI Job Portal

Takes cleaned resume text (compatible with Day 5's
`resume_extraction_engine` output) and segments it into labeled
section blocks (Skills, Experience, Education, Certifications,
Projects, ...) using a rule-based heading detector combined with an
NLP-style keyword classifier for content that has no explicit
heading, is mislabeled, or comes from a table/column layout.
"""

from .classifier import SectionClassifierEngine
from .storage import SectionBlockRecord, SectionClassificationRecord

__all__ = [
    "SectionClassifierEngine",
    "SectionBlockRecord",
    "SectionClassificationRecord",
]
__version__ = "1.0.0"
