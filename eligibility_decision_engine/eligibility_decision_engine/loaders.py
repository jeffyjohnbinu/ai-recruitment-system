"""
Alias-tolerant loaders.

Upstream ATS exports and recruiter-authored rule files vary in key
naming (camelCase vs snake_case, "score" vs "ats_score" vs
"match_score", etc). These loaders normalize via alias maps so the
rest of the engine only ever sees the canonical schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

from .schema import CandidateInput, RuleConfig

PathLike = Union[str, Path]

# --- alias maps -----------------------------------------------------

CANDIDATE_ALIASES: dict[str, list[str]] = {
    "candidate_id": ["candidate_id", "candidateId", "id", "cand_id"],
    "job_role": ["job_role", "jobRole", "role", "position", "job_title"],
    "ats_score": ["ats_score", "atsScore", "score", "match_score", "matchScore"],
    "skills": ["skills", "candidate_skills", "skill_list"],
    "experience_years": ["experience_years", "experienceYears", "experience", "years_experience"],
    "location": ["location", "candidate_location", "city"],
    "remote_ok": ["remote_ok", "remoteOk", "open_to_remote", "remote"],
    "availability": ["availability", "start_availability", "notice_period"],
}

RULE_ALIASES: dict[str, list[str]] = {
    "job_role": ["job_role", "jobRole", "role", "position"],
    "min_ats_score": ["min_ats_score", "minAtsScore", "min_score", "cutoff_score"],
    "review_band": ["review_band", "reviewBand", "review_margin"],
    "mandatory_skills": [
        "mandatory_skills",
        "mandatorySkills",
        "required_skills",
        "must_have_skills",
    ],
    "min_experience_years": ["min_experience_years", "minExperienceYears", "min_experience"],
    "max_experience_years": ["max_experience_years", "maxExperienceYears", "max_experience"],
    "allowed_locations": ["allowed_locations", "allowedLocations", "locations"],
    "allow_remote": ["allow_remote", "allowRemote", "remote_allowed"],
    "required_availability": [
        "required_availability",
        "requiredAvailability",
        "availability_required",
    ],
}


def _pick(record: dict[str, Any], aliases: list[str], default: Any = None) -> Any:
    for key in aliases:
        if key in record and record[key] is not None:
            return record[key]
    return default


def _normalize(record: dict[str, Any], alias_map: dict[str, list[str]]) -> dict[str, Any]:
    return {canonical: _pick(record, aliases) for canonical, aliases in alias_map.items()}


# --- public loaders ---------------------------------------------------


def load_ats_results(path: PathLike) -> list[CandidateInput]:
    """Load ATS output (list of candidate records) into CandidateInput list.

    Accepts either a top-level JSON list, or a dict with a
    "candidates"/"results"/"data" wrapper key (common ATS export shapes).
    """
    data = json.loads(Path(path).read_text())

    if isinstance(data, dict):
        for key in ("candidates", "results", "data", "records"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            raise ValueError(
                "load_ats_results: dict input but no recognizable list key "
                "(expected one of: candidates, results, data, records)"
            )

    if not isinstance(data, list):
        raise ValueError("load_ats_results: expected a JSON list of candidate records")

    candidates: list[CandidateInput] = []
    for raw in data:
        norm = _normalize(raw, CANDIDATE_ALIASES)
        candidates.append(
            CandidateInput(
                candidate_id=str(norm.get("candidate_id") or ""),
                job_role=str(norm.get("job_role") or ""),
                ats_score=float(norm.get("ats_score") or 0.0),
                skills=list(norm.get("skills") or []),
                experience_years=float(norm.get("experience_years") or 0.0),
                location=norm.get("location"),
                remote_ok=bool(norm.get("remote_ok") or False),
                availability=norm.get("availability"),
                raw=raw,
            )
        )
    return candidates


def load_job_rules(path: PathLike) -> dict[str, RuleConfig]:
    """Load recruiter-defined rule config(s).

    Accepts either a single rule object, or a list of rule objects
    (one per job role). Returns dict keyed by job_role for O(1) lookup.
    """
    data = json.loads(Path(path).read_text())

    if isinstance(data, dict) and "rules" in data and isinstance(data["rules"], list):
        data = data["rules"]

    records = data if isinstance(data, list) else [data]

    rules: dict[str, RuleConfig] = {}
    for raw in records:
        norm = _normalize(raw, RULE_ALIASES)
        job_role = str(norm.get("job_role") or "")
        rules[job_role] = RuleConfig(
            job_role=job_role,
            min_ats_score=float(norm.get("min_ats_score") or 0.0),
            review_band=float(norm.get("review_band") or 0.0),
            mandatory_skills=list(norm.get("mandatory_skills") or []),
            min_experience_years=float(norm.get("min_experience_years") or 0.0),
            max_experience_years=(
                float(norm["max_experience_years"])
                if norm.get("max_experience_years") is not None
                else None
            ),
            allowed_locations=list(norm.get("allowed_locations") or []),
            allow_remote=bool(
                norm.get("allow_remote") if norm.get("allow_remote") is not None else True
            ),
            required_availability=norm.get("required_availability"),
        )
    return rules
