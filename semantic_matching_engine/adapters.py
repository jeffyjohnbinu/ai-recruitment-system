"""
Input Adapters
---------------
Normalizes the many shapes prior pipeline days can hand us into a flat
{"skills": str, "experience": str, "projects": str} text dict, so
`similarity.compute_similarity` never has to know or care where the text
came from.

Every adapter function returns (sections_dict, warnings_list).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

_TARGET_SECTIONS = ("skills", "experience", "projects")

# Section-classifier heading variants (Day 8) that map onto our three
# target buckets. Kept intentionally small and explicit rather than
# importing Day 8's alias table directly, so this module has no hard
# dependency on resume_extraction_engine being importable/on the path.
_HEADING_MAP = {
    "skills": ["skills", "technical skills", "core competencies", "key skills"],
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment history",
    ],
    "projects": ["projects", "academic projects", "key projects", "personal projects"],
}


def _match_heading(key: str) -> str | None:
    key_lower = key.strip().lower()
    for target, aliases in _HEADING_MAP.items():
        if key_lower in aliases:
            return target
    return None


def _looks_like_section_map(data: dict) -> bool:
    """True if `data`'s keys look like resume section headings (Day 8
    output) rather than our own target keys or an unrelated schema."""
    return any(_match_heading(k) is not None for k in data.keys())


def _text_from_skill_records(records: List[Any]) -> str:
    """Day 9 skill_extraction_engine style: list of dicts with a
    skill/name field (and possibly confidence/synonym metadata we don't
    need here)."""
    names = []
    for item in records:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            name = item.get("skill") or item.get("name") or item.get("value")
            if name:
                names.append(str(name))
    return ", ".join(names)


def _text_from_experience_records(records: List[Any]) -> str:
    """Day 10 experience_parsing_engine style: list of dicts describing
    each role (title, company, description/responsibilities)."""
    chunks = []
    for item in records:
        if isinstance(item, str):
            chunks.append(item)
        elif isinstance(item, dict):
            pieces = [
                item.get("title") or item.get("role"),
                item.get("company") or item.get("organization"),
                item.get("description") or item.get("summary") or item.get("responsibilities"),
            ]
            chunks.append(" - ".join(str(p) for p in pieces if p))
    return "\n".join(chunks)


def normalize_resume_input(resume_input: Any) -> Tuple[Dict[str, str], List[str]]:
    warnings: List[str] = []
    sections = {k: "" for k in _TARGET_SECTIONS}

    if isinstance(resume_input, str):
        # Raw flat text (e.g. Day 5 cleaned_text). Every section gets the
        # same text -- not ideal (sections aren't isolated) but keeps the
        # engine functional, per cross-day integration discipline.
        for key in _TARGET_SECTIONS:
            sections[key] = resume_input
        warnings.append(
            "Resume input was flat text; skills/experience/projects sections "
            "were not isolated from one another."
        )
        return sections, warnings

    if isinstance(resume_input, dict):
        # Case 1: already in our target shape.
        if all(k in resume_input for k in _TARGET_SECTIONS) or set(resume_input.keys()) <= set(
            _TARGET_SECTIONS
        ):
            for key in _TARGET_SECTIONS:
                value = resume_input.get(key, "")
                sections[key] = _coerce_to_text(value)
            return sections, warnings

        # Case 2: Day 8 section-classifier map ({"Skills": "...", ...}).
        if _looks_like_section_map(resume_input):
            for key, value in resume_input.items():
                target = _match_heading(key)
                if target:
                    sections[target] = _coerce_to_text(value)
            missing = [k for k in _TARGET_SECTIONS if not sections[k]]
            if missing:
                warnings.append(f"Sections not found in resume input: {', '.join(missing)}")
            return sections, warnings

        # Case 3: a bundle with keys like "skills"/"experience"/"projects"
        # pointing at structured records (Day 9 / Day 10 outputs) rather
        # than plain strings.
        if "skills" in resume_input:
            sections["skills"] = _coerce_to_text(resume_input.get("skills"))
        if "experience" in resume_input:
            sections["experience"] = _coerce_to_text(resume_input.get("experience"))
        if "projects" in resume_input:
            sections["projects"] = _coerce_to_text(resume_input.get("projects"))

        missing = [k for k in _TARGET_SECTIONS if not sections[k]]
        if missing:
            warnings.append(f"Sections not found in resume input: {', '.join(missing)}")
        return sections, warnings

    if isinstance(resume_input, list):
        # Ambiguous flat list -- treat as skills unless items look like
        # experience entries (have title/company/description keys).
        if (
            resume_input
            and isinstance(resume_input[0], dict)
            and ("title" in resume_input[0] or "company" in resume_input[0])
        ):
            sections["experience"] = _text_from_experience_records(resume_input)
        else:
            sections["skills"] = _text_from_skill_records(resume_input)
        warnings.append("Resume input was a bare list; inferred section from record shape.")
        return sections, warnings

    warnings.append(f"Unrecognized resume input type: {type(resume_input).__name__}")
    return sections, warnings


def normalize_job_input(job_input: Any) -> Tuple[Dict[str, str], List[str]]:
    warnings: List[str] = []
    sections = {k: "" for k in _TARGET_SECTIONS}

    if isinstance(job_input, str):
        for key in _TARGET_SECTIONS:
            sections[key] = job_input
        warnings.append(
            "Job input was flat text; skills/experience/projects sections "
            "were not isolated from one another."
        )
        return sections, warnings

    if isinstance(job_input, dict):
        # Day 6 JobRequirementRecord shape: role, skills, experience, education.
        if "skills" in job_input or "experience" in job_input or "role" in job_input:
            sections["skills"] = _coerce_to_text(job_input.get("skills", ""))
            sections["experience"] = _coerce_to_text(
                job_input.get("experience") or job_input.get("experience_required", "")
            )
            # JD records rarely have a dedicated "projects" field; fall back
            # to the role/title + responsibilities text if present so the
            # projects axis still has something to compare against.
            sections["projects"] = _coerce_to_text(
                job_input.get("projects")
                or job_input.get("responsibilities")
                or job_input.get("role", "")
            )
            missing = [k for k in _TARGET_SECTIONS if not sections[k]]
            if missing:
                warnings.append(f"Sections not found in job input: {', '.join(missing)}")
            return sections, warnings

        if _looks_like_section_map(job_input):
            for key, value in job_input.items():
                target = _match_heading(key)
                if target:
                    sections[target] = _coerce_to_text(value)
            missing = [k for k in _TARGET_SECTIONS if not sections[k]]
            if missing:
                warnings.append(f"Sections not found in job input: {', '.join(missing)}")
            return sections, warnings

        warnings.append("Job input dict did not match any known schema; no sections extracted.")
        return sections, warnings

    warnings.append(f"Unrecognized job input type: {type(job_input).__name__}")
    return sections, warnings


def _coerce_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            # Could be skill records or experience records -- try both,
            # first one that yields non-empty text wins.
            as_skills = _text_from_skill_records(value)
            if as_skills.strip():
                return as_skills
            return _text_from_experience_records(value)
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return " ".join(str(v) for v in value.values() if v)
    return str(value)
