"""
Experience Parser
------------------
Extracts structured work-experience entries (company, title, dates,
bullet lines) from cleaned resume text.

Input is expected to be text that has already passed through the
Day 5 Resume Extraction Engine (`resume_extraction_engine.cleaner`),
so section headings are canonicalized (e.g. "Experience") and bullets
are normalized to "- ". Raw/uncleaned text is still handled reasonably
via the local heading-alias fallback below.

Recognized role-header line shapes (title/company separator and the
date-range separator are both tolerant of -, --, —, – and "to"):

    "Senior Backend Engineer — Acme Corp (2021 – Present)"
    "Data Scientist -- FinPay (2019-Present)"
    "Product Manager, Globex (Jan 2020 to Mar 2023)"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional

from .duration import ParsedDuration, parse_duration, yearmonth_to_iso

# Canonical "Experience" heading + common raw variants, in case the
# input text was not pre-cleaned by the Day 5 engine.
_EXPERIENCE_HEADING_ALIASES = {
    "experience",
    "work experience",
    "professional experience",
    "employment history",
    "work history",
}

# Any of these headings mark the END of the experience section when
# encountered after it has started.
_OTHER_SECTION_HEADINGS = {
    "summary",
    "professional summary",
    "career summary",
    "profile",
    "about me",
    "objective",
    "education",
    "academic background",
    "academic qualifications",
    "skills",
    "technical skills",
    "core competencies",
    "key skills",
    "areas of expertise",
    "projects",
    "academic projects",
    "key projects",
    "personal projects",
    "certifications",
    "certificates",
    "licenses & certifications",
    "licenses and certifications",
    "achievements",
    "accomplishments",
    "awards",
    "honors",
    "honors & awards",
    "languages",
    "language proficiency",
    "contact",
    "contact information",
    "contact details",
    "personal details",
    "references",
}

_SEP = r"(?:—|--|-|–|,)"
_RANGE_SEP = r"(?:–|-|to)"

_ROLE_HEADER_RE = re.compile(
    rf"""^
    (?P<title>[^()]+?)
    \s*{_SEP}\s*
    (?P<company>[^()]+?)
    \s*\(\s*
    (?P<start>[A-Za-z0-9.\s]+?)
    \s*{_RANGE_SEP}\s*
    (?P<end>[A-Za-z0-9.\s]+?)
    \s*\)\s*$
    """,
    re.VERBOSE,
)

_BULLET_RE = re.compile(r"^-\s+")


@dataclass
class ExperienceEntry:
    title: str
    company: str
    start_raw: str
    end_raw: str
    start_date: Optional[str]  # ISO "YYYY-MM" or None if unparseable
    end_date: Optional[str]
    duration_months: int
    is_current: bool
    date_parse_ok: bool
    bullet_lines: List[str] = field(default_factory=list)
    raw_line: str = ""

    @property
    def description(self) -> str:
        return " ".join(self.bullet_lines)


@dataclass
class ExperienceParseResult:
    entries: List[ExperienceEntry]
    lines_matched: int
    lines_unmatched_in_section: List[str]
    section_found: bool


class ExperienceParser:
    """Extracts ExperienceEntry objects from cleaned resume text."""

    def __init__(self, today: Optional[date] = None):
        self._today = today

    def parse(self, text: str) -> ExperienceParseResult:
        lines = text.splitlines()
        entries: List[ExperienceEntry] = []
        unmatched: List[str] = []
        current: Optional[ExperienceEntry] = None

        in_experience_section = False
        section_found = False
        # If no section headings exist at all in the text, treat the
        # whole document as fair game (handles isolated snippets / tests).
        has_any_heading = any(
            line.strip().strip(":").lower() in _EXPERIENCE_HEADING_ALIASES
            or line.strip().strip(":").lower() in _OTHER_SECTION_HEADINGS
            for line in lines
        )

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            heading_key = line.strip(":").lower()

            if heading_key in _EXPERIENCE_HEADING_ALIASES:
                in_experience_section = True
                section_found = True
                continue

            if heading_key in _OTHER_SECTION_HEADINGS:
                if in_experience_section:
                    in_experience_section = False
                continue

            active = in_experience_section or not has_any_heading
            if not active:
                continue

            role_match = _ROLE_HEADER_RE.match(line)
            if role_match:
                entry = self._build_entry(role_match, raw_line=line)
                entries.append(entry)
                current = entry
                continue

            if _BULLET_RE.match(line):
                if current is not None:
                    current.bullet_lines.append(_BULLET_RE.sub("", line).strip())
                continue

            # Non-bullet, non-role, non-heading line inside the section:
            # could be a name/contact header, a stray descriptive line,
            # or an unrecognized role-header format. Track it but don't
            # attach it to an entry.
            if in_experience_section or not has_any_heading:
                unmatched.append(line)

        return ExperienceParseResult(
            entries=entries,
            lines_matched=len(entries),
            lines_unmatched_in_section=unmatched,
            section_found=section_found,
        )

    def _build_entry(self, match: "re.Match[str]", raw_line: str) -> ExperienceEntry:
        title = match.group("title").strip()
        company = match.group("company").strip()
        start_raw = match.group("start").strip()
        end_raw = match.group("end").strip()

        parsed: ParsedDuration = parse_duration(start_raw, end_raw, today=self._today)

        return ExperienceEntry(
            title=title,
            company=company,
            start_raw=start_raw,
            end_raw=end_raw,
            start_date=yearmonth_to_iso(parsed.start),
            end_date=yearmonth_to_iso(parsed.end),
            duration_months=parsed.months,
            is_current=parsed.is_current,
            date_parse_ok=parsed.parse_ok,
            raw_line=raw_line,
        )
