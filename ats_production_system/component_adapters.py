"""
component_adapters.py
-----------------------
Day 20 deliverable — Zecpath AI Job Portal

This module exists because of a specific finding from the Day 20 final
review: the Day 17 test harness's `pipeline_adapter.py` guesses at import
paths that don't match the real Day 12-14 modules (e.g.
`ats_scoring_engine.scorer.score_candidate`, which has never existed —
the real class is `ats_scoring_engine.scoring_engine.ATSScoringEngine`).
Because of that, every Day 17 test run silently fell through to the
keyword-overlap fallback scorer. The reported 87.5% accuracy number was
therefore measuring the *fallback*, not the real semantic/scoring
pipeline (see ats_system_testing's own PERF-001 backlog note for the
related, but distinct, "not importable outside the integrated repo"
caveat).

A second, deeper issue this module fixes: even when the real engines
*are* imported correctly, their output field names don't line up with
what `ats_scoring_engine.loaders` looks for:

    Semantic Matching Engine  -> `overall_score`      but loaders want
                                                          `similarity_score`
    Experience Relevance      -> `overall_relevance_score` but loaders
                                                          want `relevance_score`
    Education/Certification   -> raw degree/cert lists, no alignment
                                  score or boolean signals against a
                                  target job at all
    Skill Extraction          -> a flat skill list, not a
                                  matched/required pair against a JD

None of the upstream Day 9-13 modules are modified to fix this (additive-
only integration discipline). Instead, this adapter layer sits between
them and the scoring engine and performs the real field mapping /
derivation, so `ats_scoring_engine.compute_score()` actually receives
data it can use instead of silently reporting every component
"unavailable" and falling back to renormalized/neutral scoring.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _normalize_skill_name(name: str) -> str:
    return name.strip().lower()


def build_skill_component(
    candidate_skills: List[str],
    job_required_skills: List[str],
    job_preferred_skills: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Maps Day 9 (SkillExtractionEngine) output + Day 6 (JD) required/preferred
    skills into the `matched_skills` / `required_skills` shape that
    `ats_scoring_engine.loaders.extract_skill_match` actually recognizes.
    """
    job_preferred_skills = job_preferred_skills or []
    all_required = list(dict.fromkeys(job_required_skills + job_preferred_skills))

    candidate_norm = {_normalize_skill_name(s) for s in candidate_skills}
    required_norm = [_normalize_skill_name(s) for s in all_required]

    matched = [s for s in all_required if _normalize_skill_name(s) in candidate_norm]

    return {
        "matched_skills": matched,
        "required_skills": all_required,
        "_candidate_skill_count": len(candidate_skills),
        "_required_skill_count": len(all_required),
        "_normalized_candidate_skills": sorted(candidate_norm),
        "_normalized_required_skills": required_norm,
    }


def build_experience_component(relevance_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Maps Day 10's ExperienceRelevanceRecord.to_dict() (which uses
    `overall_relevance_score`) onto `relevance_score`, the field name
    `extract_experience_relevance` actually looks for.
    """
    return {
        "relevance_score": relevance_record.get("overall_relevance_score"),
        "relevant_experience_years": relevance_record.get("relevant_experience_years"),
        "total_experience_years": relevance_record.get("total_experience_years"),
    }


_DEGREE_RANK_ORDER = [
    "phd",
    "doctorate",
    "master",
    "mba",
    "bachelor",
    "associate",
    "diploma",
    "high school diploma",
]


def _degree_rank(label: Optional[str]) -> int:
    if not label:
        return len(_DEGREE_RANK_ORDER)
    label_lower = label.lower()
    for i, key in enumerate(_DEGREE_RANK_ORDER):
        if key in label_lower:
            return i
    return len(_DEGREE_RANK_ORDER)


def build_education_component(
    academic_profile: Dict[str, Any],
    job_education_level: Optional[str],
    job_education_fields: Optional[List[str]],
) -> Dict[str, Any]:
    """
    Day 11's AcademicProfileRecord carries a raw degree/certification list
    and a `highest_degree` string, but no alignment score or boolean
    signal against a target job at all -- `extract_education_alignment`
    can only use `meets_minimum_education` / `field_match` booleans (or a
    direct score field), so this adapter derives them:

      meets_minimum_education: candidate's highest degree rank is at or
        above the JD's stated minimum (by a small hand-ranked ladder --
        PhD > Master's/MBA > Bachelor's > Associate > Diploma/HS).
        If the JD states no education requirement, this is treated as met.

      field_match: whether any degree's field-of-study text overlaps
        (word-level) with any of the JD's stated education_fields.
        If the JD lists no field, this is treated as met (no constraint).
    """
    highest_degree = academic_profile.get("highest_degree")
    degrees = academic_profile.get("degrees") or []
    certifications = academic_profile.get("certifications") or []

    if job_education_level:
        meets_minimum = _degree_rank(highest_degree) <= _degree_rank(job_education_level)
    else:
        meets_minimum = True

    if job_education_fields:
        candidate_fields = set()
        for d in degrees:
            field = (d.get("field") or d.get("field_of_study") or "") if isinstance(d, dict) else ""
            candidate_fields.update(w.lower() for w in field.split() if len(w) > 2)
        jd_field_tokens = set()
        for f in job_education_fields:
            jd_field_tokens.update(w.lower() for w in f.split() if len(w) > 2)
        field_match = bool(candidate_fields & jd_field_tokens) if jd_field_tokens else True
    else:
        field_match = True

    return {
        "meets_minimum_education": meets_minimum,
        "field_match": field_match,
        "relevant_certifications": certifications,
        "highest_degree": highest_degree,
    }


def build_semantic_component(semantic_match_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Maps Day 12's SemanticMatchRecord.to_dict() (`overall_score`) onto
    `similarity_score`, the field name `extract_semantic_similarity`
    actually looks for.
    """
    return {
        "similarity_score": semantic_match_record.get("overall_score"),
        "match_band": semantic_match_record.get("match_band"),
        "section_scores": semantic_match_record.get("section_scores"),
    }
