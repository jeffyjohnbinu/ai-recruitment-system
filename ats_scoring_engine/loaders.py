"""
loaders.py
----------
Alias-tolerant loaders that pull a single normalized 0-1 component score
out of upstream module outputs:

    Day 9  -> Skill Extraction Engine            -> skill_match
    Day 10 -> Experience Parsing & Relevance      -> experience_relevance
    Day 11 -> Education & Certification Parsing   -> education_alignment
    Day 12 -> Semantic Matching Engine            -> semantic_similarity

Following the additive-only integration convention established across the
pipeline, these loaders never assume a single rigid schema. They try a list
of known field-name aliases (flat and nested), and fall back to deriving a
ratio from raw lists/counts when no direct numeric field is present. If
nothing usable is found, the component is reported as unavailable rather
than raising -- callers (ATSScoringEngine) decide how to handle missing
data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class ComponentExtractionResult:
    score: Optional[float]
    available: bool
    source_fields_used: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def _get_path(data: Any, path: str) -> Any:
    """Resolve a dotted path (e.g. 'result.summary.score') in a nested dict."""
    cur = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _first_numeric(data: dict, paths: List[str]) -> Optional[tuple[float, str]]:
    for p in paths:
        val = _get_path(data, p)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val), p
    return None


def _normalize_to_unit(value: float) -> float:
    """Scale a plausible percentage (>1) down to [0, 1] and clamp."""
    if value > 1.0:
        value = value / 100.0
    return max(0.0, min(1.0, value))


def _ratio_from_lists(data: dict, matched_paths: List[str], total_paths: List[str]):
    for mp in matched_paths:
        matched = _get_path(data, mp)
        if isinstance(matched, list):
            for tp in total_paths:
                total = _get_path(data, tp)
                if isinstance(total, list) and len(total) > 0:
                    return len(matched) / len(total), f"{mp} / {tp} ratio"
    return None


# --------------------------------------------------------------------- #
# Day 9 -- Skill Extraction Engine
# --------------------------------------------------------------------- #
def extract_skill_match(skill_data: Optional[dict]) -> ComponentExtractionResult:
    if not skill_data:
        return ComponentExtractionResult(None, False, notes=["skill extraction data not provided"])

    direct_paths = [
        "skill_match_score",
        "skill_match_percentage",
        "match_percentage",
        "match_score",
        "result.skill_match_score",
        "result.match_percentage",
        "summary.match_percentage",
        "summary.skill_match_score",
        "skills.match_score",
    ]
    found = _first_numeric(skill_data, direct_paths)
    if found:
        val, source = found
        return ComponentExtractionResult(_normalize_to_unit(val), True, source_fields_used=[source])

    ratio = _ratio_from_lists(
        skill_data,
        matched_paths=["matched_skills", "result.matched_skills", "skills.matched"],
        total_paths=["required_skills", "result.required_skills", "skills.required", "jd_skills"],
    )
    if ratio:
        val, source = ratio
        return ComponentExtractionResult(_normalize_to_unit(val), True, source_fields_used=[source])

    return ComponentExtractionResult(
        None,
        False,
        notes=["no recognizable skill-match field or matched/required skill lists found"],
    )


# --------------------------------------------------------------------- #
# Day 10 -- Experience Parsing & Relevance Engine
# --------------------------------------------------------------------- #
def extract_experience_relevance(experience_data: Optional[dict]) -> ComponentExtractionResult:
    if not experience_data:
        return ComponentExtractionResult(
            None, False, notes=["experience parsing data not provided"]
        )

    direct_paths = [
        "relevance_score",
        "experience_relevance",
        "experience_relevance_score",
        "result.relevance_score",
        "result.experience_relevance",
        "summary.relevance_score",
    ]
    found = _first_numeric(experience_data, direct_paths)
    if found:
        val, source = found
        return ComponentExtractionResult(_normalize_to_unit(val), True, source_fields_used=[source])

    # Fallback: derive from years-of-relevant-experience vs. years-required
    years_paths = [
        "relevant_years",
        "years_relevant",
        "result.relevant_years",
        "total_relevant_years",
    ]
    required_years_paths = [
        "required_years",
        "years_required",
        "result.required_years",
        "jd_required_years",
    ]
    yr_found = _first_numeric(experience_data, years_paths)
    req_found = _first_numeric(experience_data, required_years_paths)
    if yr_found and req_found and req_found[0] > 0:
        ratio = yr_found[0] / req_found[0]
        return ComponentExtractionResult(
            _normalize_to_unit(ratio),
            True,
            source_fields_used=[f"{yr_found[1]} / {req_found[1]} ratio"],
        )

    return ComponentExtractionResult(
        None, False, notes=["no recognizable experience relevance field or year ratio found"]
    )


# --------------------------------------------------------------------- #
# Day 11 -- Education & Certification Parsing
# --------------------------------------------------------------------- #
def extract_education_alignment(education_data: Optional[dict]) -> ComponentExtractionResult:
    if not education_data:
        return ComponentExtractionResult(None, False, notes=["education parsing data not provided"])

    direct_paths = [
        "education_match_score",
        "education_alignment_score",
        "degree_alignment_score",
        "result.education_match_score",
        "summary.education_match_score",
    ]
    found = _first_numeric(education_data, direct_paths)
    if found:
        val, source = found
        return ComponentExtractionResult(_normalize_to_unit(val), True, source_fields_used=[source])

    # Fallback: combine boolean-ish signals (degree met + field match + certs bonus)
    degree_met = _get_path(education_data, "meets_minimum_education")
    field_match = _get_path(education_data, "field_match")
    cert_bonus = education_data.get("relevant_certifications")

    if isinstance(degree_met, bool) or isinstance(field_match, bool):
        score = 0.0
        parts_used = 0
        if isinstance(degree_met, bool):
            score += 0.6 if degree_met else 0.0
            parts_used += 1
        if isinstance(field_match, bool):
            score += 0.4 if field_match else 0.0
            parts_used += 1
        if isinstance(cert_bonus, list) and cert_bonus:
            score = min(1.0, score + 0.1)
        if parts_used:
            return ComponentExtractionResult(
                _normalize_to_unit(score),
                True,
                source_fields_used=["meets_minimum_education/field_match composite"],
            )

    return ComponentExtractionResult(
        None, False, notes=["no recognizable education alignment field or boolean signals found"]
    )


# --------------------------------------------------------------------- #
# Day 12 -- Semantic Matching Engine
# --------------------------------------------------------------------- #
def extract_semantic_similarity(semantic_data: Optional[dict]) -> ComponentExtractionResult:
    if not semantic_data:
        return ComponentExtractionResult(None, False, notes=["semantic matching data not provided"])

    direct_paths = [
        "similarity_score",
        "semantic_score",
        "semantic_similarity",
        "cosine_similarity",
        "result.similarity_score",
        "summary.similarity_score",
    ]
    found = _first_numeric(semantic_data, direct_paths)
    if found:
        val, source = found
        # Cosine similarity can range [-1, 1]; rescale before clamping to [0, 1]
        if -1.0 <= val < 0.0:
            val = (val + 1.0) / 2.0
        return ComponentExtractionResult(_normalize_to_unit(val), True, source_fields_used=[source])

    return ComponentExtractionResult(
        None, False, notes=["no recognizable semantic similarity field found"]
    )
