"""
Rule-based section detection
-----------------------------
Canonical section labels and the heading-alias table used to spot an
*explicit* section heading in resume text. This mirrors (and stays
compatible with) the `_SECTION_ALIASES` table in
`resume_extraction_engine/cleaner.py` from Day 5, so a heading that
Day 5 has already canonicalized (e.g. "WORK EXPERIENCE" -> "Experience")
is recognized here with zero extra work, while this module can also
run standalone on raw/uncleaned text.

Two extra labels exist beyond Day 5's set, because segmentation (this
module's job) needs to bucket content Day 5 never had to classify:
  - "Header"   the name/contact block at the very top, before any
               section heading appears
  - "Uncategorized"  content the classifier genuinely can't place
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

# Canonical section labels this module can assign.
CANONICAL_LABELS: List[str] = [
    "Header",
    "Contact",
    "Summary",
    "Experience",
    "Education",
    "Skills",
    "Projects",
    "Certifications",
    "Achievements",
    "Languages",
    "References",
    "Uncategorized",
]

# Heading text variants -> canonical label. Kept in sync with Day 5's
# `_SECTION_ALIASES` for Skills/Experience/Education/Certifications/
# Projects/Achievements/Languages/References/Summary, plus a couple of
# extra phrasings this module also treats as headings.
SECTION_ALIASES: Dict[str, List[str]] = {
    "Summary": [
        "summary",
        "professional summary",
        "career summary",
        "profile",
        "about me",
        "objective",
        "career objective",
    ],
    "Contact": [
        "contact",
        "contact information",
        "contact details",
        "personal details",
    ],
    "Experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "work history",
        "relevant experience",
    ],
    "Education": [
        "education",
        "academic background",
        "academic qualifications",
        "educational qualifications",
    ],
    "Skills": [
        "skills",
        "technical skills",
        "core competencies",
        "key skills",
        "areas of expertise",
        "skills & tools",
        "skills and tools",
    ],
    "Projects": ["projects", "academic projects", "key projects", "personal projects"],
    "Certifications": [
        "certifications",
        "certificates",
        "licenses & certifications",
        "licenses and certifications",
    ],
    "Achievements": ["achievements", "accomplishments", "awards", "honors", "honors & awards"],
    "Languages": ["languages", "language proficiency"],
    "References": ["references"],
}

HEADING_LOOKUP: Dict[str, str] = {
    alias.lower(): canonical for canonical, aliases in SECTION_ALIASES.items() for alias in aliases
}

_MAX_HEADING_LEN = 40
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"(\+?\d[\d\-\s()]{6,}\d)")


def match_heading(line: str) -> Optional[str]:
    """
    Return the canonical section label if `line` is (or looks like) a
    section heading, else None. Handles headings Day 5 already
    canonicalized ("Experience") as well as raw variants ("WORK
    EXPERIENCE", "technical skills:", "Key Projects").
    """
    stripped = line.strip().rstrip(":").strip()
    if not stripped or len(stripped) > _MAX_HEADING_LEN:
        return None

    key = stripped.lower()
    if key in HEADING_LOOKUP:
        return HEADING_LOOKUP[key]

    return None


def looks_like_contact_line(line: str) -> bool:
    """True if a line carries an email or a phone-number-shaped token."""
    return bool(_EMAIL_RE.search(line)) or bool(_PHONE_RE.search(line))
