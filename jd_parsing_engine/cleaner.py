"""
Cleaning & Normalization for Job Descriptions
-----------------------------------------------
Turns raw job-description text (pasted from a job board, ATS export, or
Word/PDF posting) into a clean, normalized text block ready for
structured field extraction.

Cleaning:
  - strips control characters, weird unicode artifacts, stray symbols
  - collapses excess whitespace / blank lines
  - de-hyphenates line-wrapped words
  - drops boilerplate noise lines (EEO statements, page numbers, apply-now
    footers) that add no signal for requirement extraction

Normalization:
  - unifies bullet characters into a single "- " style
  - normalizes section headings (e.g. "What You'll Need" / "Requirements"
    / "Qualifications" -> canonical "Requirements") so downstream
    extractors can reliably scope their search to the right section
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

_BULLET_CHARS = "•●○◦▪▫■□‣∙·‐‑–—*➤➔→»›"
_BULLET_PATTERN = re.compile(rf"^[\s]*[{re.escape(_BULLET_CHARS)}]\s*")
_CID_BULLET_PATTERN = re.compile(r"^\s*\(cid:\d+\)\s*")

_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE_PATTERN = re.compile(r"[ \t]{2,}")
_MULTI_BLANK_LINE_PATTERN = re.compile(r"\n{3,}")
_HYPHEN_WRAP_PATTERN = re.compile(r"(\w)-\n(\w)")
_PAGE_NUM_PATTERN = re.compile(r"^\s*(page\s*)?\d+\s*(of\s*\d+)?\s*$", re.IGNORECASE)
_NON_PRINTABLE_UNICODE = re.compile(r"[\uf000-\uf8ff\u200b\u200c\u200e\u200f\ufeff]")

# Boilerplate lines that carry no extractable requirement signal —
# dropped so they don't pollute skill/section detection.
_NOISE_LINE_PATTERNS = [
    re.compile(r"^\s*equal opportunity employer", re.IGNORECASE),
    re.compile(r"^\s*we are an equal opportunity", re.IGNORECASE),
    re.compile(r"^\s*apply (now|today|here)\b", re.IGNORECASE),
    re.compile(r"^\s*click (here|the link) to apply", re.IGNORECASE),
    re.compile(r"^\s*job (id|reference|req(uisition)? #?)\s*[:#]", re.IGNORECASE),
    re.compile(r"^\s*posted\s+(on\s+)?\d", re.IGNORECASE),
]

# Canonical JD section headings + common variants seen across job boards
# (LinkedIn, Indeed, Greenhouse, Lever, Workday exports, plain-text pastes)
_SECTION_ALIASES = {
    "Role Overview": [
        "role overview",
        "about the role",
        "job summary",
        "position summary",
        "overview",
        "the role",
        "about this role",
    ],
    "Responsibilities": [
        "responsibilities",
        "key responsibilities",
        "what you'll do",
        "what you will do",
        "duties",
        "role and responsibilities",
        "job duties",
    ],
    "Requirements": [
        "requirements",
        "qualifications",
        "minimum qualifications",
        "required qualifications",
        "what you'll need",
        "what you will need",
        "must haves",
        "must-haves",
        "basic qualifications",
        "who you are",
    ],
    "Preferred Qualifications": [
        "preferred qualifications",
        "nice to have",
        "nice to haves",
        "bonus points",
        "preferred skills",
        "good to have",
    ],
    "Skills": [
        "skills",
        "technical skills",
        "required skills",
        "core skills",
        "tech stack",
    ],
    "Education": [
        "education",
        "education requirements",
        "academic requirements",
        "educational qualifications",
    ],
    "Experience": [
        "experience",
        "experience requirements",
        "years of experience",
    ],
    "Benefits": [
        "benefits",
        "perks",
        "what we offer",
        "compensation and benefits",
        "why join us",
    ],
    "Company": [
        "about us",
        "about the company",
        "who we are",
        "company overview",
    ],
}

_SECTION_LOOKUP = {
    alias.lower(): canonical for canonical, aliases in _SECTION_ALIASES.items() for alias in aliases
}


@dataclass
class CleaningReport:
    removed_control_chars: int = 0
    collapsed_blank_lines: int = 0
    normalized_bullets: int = 0
    normalized_headings: int = 0
    dehyphenated_words: int = 0
    dropped_noise_lines: List[str] = field(default_factory=list)


class JDTextCleaner:
    """Applies noise removal and normalization to raw job description text."""

    def clean(self, raw_text: str) -> tuple[str, CleaningReport]:
        report = CleaningReport()
        text = raw_text

        # 1. Strip control chars & odd unicode
        cleaned, n = _CONTROL_CHAR_PATTERN.subn("", text)
        report.removed_control_chars += n
        text = _NON_PRINTABLE_UNICODE.sub("", cleaned)

        # 2. Fix line-wrap hyphenation
        text, n = _HYPHEN_WRAP_PATTERN.subn(r"\1\2", text)
        report.dehyphenated_words = n

        # 3. Normalize per-line: bullets, headings, drop noise
        lines_out: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()

            if not stripped:
                lines_out.append("")
                continue

            if _PAGE_NUM_PATTERN.match(stripped) or any(
                p.match(stripped) for p in _NOISE_LINE_PATTERNS
            ):
                report.dropped_noise_lines.append(stripped)
                continue

            if _BULLET_PATTERN.match(stripped):
                stripped = "- " + _BULLET_PATTERN.sub("", stripped).strip()
                report.normalized_bullets += 1
            elif _CID_BULLET_PATTERN.match(stripped):
                stripped = "- " + _CID_BULLET_PATTERN.sub("", stripped).strip()
                report.normalized_bullets += 1

            heading_key = stripped.strip(":").strip().lower()
            if heading_key in _SECTION_LOOKUP and len(stripped) < 50:
                stripped = _SECTION_LOOKUP[heading_key]
                report.normalized_headings += 1

            lines_out.append(stripped)

        text = "\n".join(lines_out)

        # 4. Collapse whitespace / blank-line runs
        text = _MULTI_SPACE_PATTERN.sub(" ", text)
        text, n = _MULTI_BLANK_LINE_PATTERN.subn("\n\n", text)
        report.collapsed_blank_lines = n

        text = text.strip() + "\n"
        return text, report

    @staticmethod
    def detect_sections(cleaned_text: str) -> List[str]:
        canonical_headings = set(_SECTION_ALIASES.keys())
        found: List[str] = []
        for line in cleaned_text.splitlines():
            stripped = line.strip()
            if stripped in canonical_headings and stripped not in found:
                found.append(stripped)
        return found

    @staticmethod
    def section_text(cleaned_text: str, section_name: str) -> str:
        """
        Return the text belonging to one canonical section only, i.e. the
        lines between this section's heading and the next recognized
        heading (or end of document).
        """
        canonical_headings = set(_SECTION_ALIASES.keys())
        lines = cleaned_text.splitlines()
        collecting = False
        out: List[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped in canonical_headings:
                if stripped == section_name:
                    collecting = True
                    continue
                elif collecting:
                    break
            elif collecting:
                out.append(line)
        return "\n".join(out)
