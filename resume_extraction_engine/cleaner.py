"""
Cleaning & Normalization
-------------------------
Turns raw extracted text (noisy, inconsistently formatted) into a clean,
normalized text block ready for downstream AI parsing.

Cleaning:
  - strips control characters, weird unicode artifacts, stray symbols
  - collapses excess whitespace / blank lines
  - de-hyphenates line-wrapped words ("develop-\nment" -> "development")
  - removes page-artifact noise (page numbers, repeated headers/footers)

Normalization:
  - unifies bullet characters (•, ●, ▪, -, *, ➤ ...) into a single "- " style
  - normalizes section headings (EXPERIENCE / Experience / experience: -> "Experience")
  - normalizes capitalization of headings without shouting the whole resume
  - trims trailing punctuation noise, normalizes whitespace around bullets
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

# Characters commonly used as bullets across PDF/DOCX exports
_BULLET_CHARS = "•●○◦▪▫■□‣∙·‐‑–—*➤➔→»›"
_BULLET_PATTERN = re.compile(rf"^[\s]*[{re.escape(_BULLET_CHARS)}]\s*")

# Some PDF fonts lack a proper ToUnicode CMap for bullet glyphs, so
# extraction libraries surface them as literal "(cid:127)"-style tokens
# instead of a real bullet character. Treat a leading cid token as a bullet.
_CID_BULLET_PATTERN = re.compile(r"^\s*\(cid:\d+\)\s*")

_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE_PATTERN = re.compile(r"[ \t]{2,}")
_MULTI_BLANK_LINE_PATTERN = re.compile(r"\n{3,}")
_HYPHEN_WRAP_PATTERN = re.compile(r"(\w)-\n(\w)")
_PAGE_NUM_PATTERN = re.compile(r"^\s*(page\s*)?\d+\s*(of\s*\d+)?\s*$", re.IGNORECASE)
_NON_PRINTABLE_UNICODE = re.compile(r"[\uf000-\uf8ff\u200b\u200c\u200e\u200f\ufeff]")

# Canonical section headings + common variants seen across resume templates
_SECTION_ALIASES = {
    "Summary": [
        "summary",
        "professional summary",
        "career summary",
        "profile",
        "about me",
        "objective",
    ],
    "Experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "work history",
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
    "Contact": ["contact", "contact information", "contact details", "personal details"],
    "References": ["references"],
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


class TextCleaner:
    """Applies noise removal and normalization to raw extracted resume text."""

    def clean(self, raw_text: str) -> tuple[str, CleaningReport]:
        report = CleaningReport()
        text = raw_text

        # 1. Strip control chars & odd unicode
        cleaned, n = _CONTROL_CHAR_PATTERN.subn("", text)
        report.removed_control_chars += n
        text = _NON_PRINTABLE_UNICODE.sub("", cleaned)

        # 2. Fix line-wrap hyphenation ("soft-\nware" -> "software")
        text, n = _HYPHEN_WRAP_PATTERN.subn(r"\1\2", text)
        report.dehyphenated_words = n

        # 3. Normalize per-line: bullets, headings, drop page-number noise
        lines_out: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()

            if not stripped:
                lines_out.append("")
                continue

            if _PAGE_NUM_PATTERN.match(stripped):
                report.dropped_noise_lines.append(stripped)
                continue

            # Bullet normalization (real bullet glyphs)
            if _BULLET_PATTERN.match(stripped):
                stripped = "- " + _BULLET_PATTERN.sub("", stripped).strip()
                report.normalized_bullets += 1
            # Bullet normalization (unrenderable glyph -> literal cid token)
            elif _CID_BULLET_PATTERN.match(stripped):
                stripped = "- " + _CID_BULLET_PATTERN.sub("", stripped).strip()
                report.normalized_bullets += 1

            # Section heading normalization
            heading_key = stripped.strip(":").strip().lower()
            if heading_key in _SECTION_LOOKUP and len(stripped) < 40:
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


class TextNormalizer:
    """
    Additional normalization pass focused on consistent capitalization,
    applied AFTER cleaning. Keeps ALL-CAPS acronyms (AWS, SQL, PMP) intact
    while fixing ALL-CAPS shouting lines that are really just headings or
    sentences typed in caps by the resume author.
    """

    _ACRONYM_SAFE_LEN = 5  # words <= this length in all-caps are likely acronyms, leave alone

    # Common short function words that must NOT be preserved as "acronyms"
    # even though they're <= 3 letters (e.g. "OF", "AND", "THE" in a
    # shouted sentence should become "of", "and", "the", not stay upper).
    _MINOR_WORDS = {
        "of",
        "and",
        "the",
        "in",
        "on",
        "at",
        "to",
        "by",
        "or",
        "an",
        "is",
        "as",
        "it",
        "if",
        "be",
        "so",
        "no",
        "up",
        "us",
        "my",
        "we",
        "a",
        "for",
        "with",
        "from",
        "into",
        "than",
        "via",
    }

    def normalize_capitalization(self, text: str) -> str:
        out_lines = []
        for line in text.splitlines():
            out_lines.append(self._normalize_line(line))
        return "\n".join(out_lines)

    def _normalize_line(self, line: str) -> str:
        stripped = line.strip()
        if not stripped:
            return line

        letters = [c for c in stripped if c.isalpha()]
        if not letters:
            return line

        is_all_caps = stripped.upper() == stripped and any(c.isalpha() for c in stripped)
        word_count = len(stripped.split())

        # Leave short all-caps tokens alone (acronyms, headings already canonicalized)
        if is_all_caps and word_count > self._ACRONYM_SAFE_LEN:
            # Title-case shouted sentences/headings, but keep known acronyms
            return self._smart_title_case(stripped)

        return line

    def _smart_title_case(self, text: str) -> str:
        words = text.split()
        result = []
        for w in words:
            core = re.sub(r"[^A-Za-z]", "", w)
            if core.lower() in self._MINOR_WORDS:
                result.append(w.lower())
            elif len(core) <= 3 and core.isalpha():
                # keep very short remaining tokens upper (likely acronym: AI, ML, PMP)
                result.append(w)
            else:
                result.append(w.capitalize())
        return " ".join(result)
