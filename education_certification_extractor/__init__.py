"""
Education & Certification Parsing Engine
Day 11 deliverable — Zecpath AI Job Portal

Detects academic qualifications (degree type, field of study, institution,
graduation year) and professional certifications from resume text, then
normalizes naming conventions and tags certifications with relevance
categories.

Consumes either:
  - a raw cleaned resume text block (as produced by Day 5's
    ResumeExtractionEngine), from which the Education/Certifications
    sections are located internally, or
  - a pre-segmented section dict (as produced by Day 8's
    resume_section_classifier), e.g. {"Education": "...", "Certifications": "..."}

Either input path produces the same AcademicProfileRecord shape, so this
module has no hard runtime dependency on Day 5 or Day 8 packages -- it just
follows the same canonical section-name and text conventions they use.
"""

from .extractor import EducationCertificationExtractor

__all__ = ["EducationCertificationExtractor"]
__version__ = "1.0.0"
