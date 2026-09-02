"""
Eligibility decision logic.

Status assignment order:
  1. Hard rule failures (mandatory skills, experience bounds,
     location/remote, availability) -> REJECTED, unless the failure
     is "soft" (see review_band note below) in which case -> REVIEW.
  2. Score check against min_ats_score:
       score >= min_ats_score                       -> ELIGIBLE (if rules pass)
       min_ats_score - review_band <= score < min    -> REVIEW
       score < min_ats_score - review_band           -> REJECTED
  3. If every check passes -> ELIGIBLE.

Any single hard-rule failure forces at best REVIEW, never ELIGIBLE,
regardless of score.
"""

from __future__ import annotations

from .schema import (
    CandidateEligibilityResult,
    CandidateInput,
    EligibilityStatus,
    RuleConfig,
    RuleReason,
)


def _check_skills(c: CandidateInput, r: RuleConfig) -> RuleReason:
    if not r.mandatory_skills:
        return RuleReason("mandatory_skills", True, "no mandatory skills configured")
    have = {s.strip().lower() for s in c.skills}
    need = {s.strip().lower() for s in r.mandatory_skills}
    missing = need - have
    if missing:
        return RuleReason("mandatory_skills", False, f"missing: {', '.join(sorted(missing))}")
    return RuleReason("mandatory_skills", True, "all mandatory skills present")


def _check_experience(c: CandidateInput, r: RuleConfig) -> RuleReason:
    if c.experience_years < r.min_experience_years:
        return RuleReason(
            "experience_range",
            False,
            f"{c.experience_years}y < min {r.min_experience_years}y",
        )
    if r.max_experience_years is not None and c.experience_years > r.max_experience_years:
        return RuleReason(
            "experience_range",
            False,
            f"{c.experience_years}y > max {r.max_experience_years}y",
        )
    return RuleReason("experience_range", True, f"{c.experience_years}y within range")


def _check_location(c: CandidateInput, r: RuleConfig) -> RuleReason:
    if c.remote_ok and r.allow_remote:
        return RuleReason("location", True, "remote candidate, remote allowed")
    if not r.allowed_locations:
        return RuleReason("location", True, "no location restriction configured")
    if c.location and c.location.strip().lower() in {
        loc.strip().lower() for loc in r.allowed_locations
    }:
        return RuleReason("location", True, f"{c.location} in allowed locations")
    return RuleReason(
        "location",
        False,
        f"location '{c.location}' not in allowed set and remote not accepted",
    )


def _check_availability(c: CandidateInput, r: RuleConfig) -> RuleReason:
    if r.required_availability is None:
        return RuleReason("availability", True, "no availability constraint configured")
    if (c.availability or "").strip().lower() == r.required_availability.strip().lower():
        return RuleReason("availability", True, f"matches required '{r.required_availability}'")
    return RuleReason(
        "availability",
        False,
        f"'{c.availability}' does not match required '{r.required_availability}'",
    )


def _check_score(c: CandidateInput, r: RuleConfig) -> RuleReason:
    if c.ats_score >= r.min_ats_score:
        return RuleReason("ats_score", True, f"{c.ats_score} >= min {r.min_ats_score}")
    floor = r.min_ats_score - r.review_band
    if c.ats_score >= floor:
        return RuleReason(
            "ats_score",
            False,
            f"{c.ats_score} below min {r.min_ats_score} but within review band (floor {floor})",
        )
    return RuleReason("ats_score", False, f"{c.ats_score} below review floor {floor}")


def evaluate_candidate(
    candidate: CandidateInput, rules: dict[str, RuleConfig]
) -> CandidateEligibilityResult:
    """Evaluate one candidate against the rule set for their job_role."""
    rule = rules.get(candidate.job_role)
    if rule is None:
        return CandidateEligibilityResult(
            candidate_id=candidate.candidate_id,
            job_role=candidate.job_role,
            status=EligibilityStatus.REVIEW,
            ats_score=candidate.ats_score,
            reasons=[
                RuleReason("job_role_config", False, "no rule config found for this job_role")
            ],
        )

    checks = [
        _check_skills(candidate, rule),
        _check_experience(candidate, rule),
        _check_location(candidate, rule),
        _check_availability(candidate, rule),
    ]
    score_check = _check_score(candidate, rule)
    checks.append(score_check)

    hard_failures = [c for c in checks[:-1] if not c.passed]  # skills/exp/loc/avail

    if hard_failures:
        status = EligibilityStatus.REJECTED
    elif not score_check.passed:
        # score_check failed but within review band -> REVIEW; below floor -> REJECTED
        status = (
            EligibilityStatus.REVIEW
            if "review band" in score_check.detail
            else EligibilityStatus.REJECTED
        )
    else:
        status = EligibilityStatus.ELIGIBLE

    return CandidateEligibilityResult(
        candidate_id=candidate.candidate_id,
        job_role=candidate.job_role,
        status=status,
        ats_score=candidate.ats_score,
        reasons=checks,
    )


def evaluate_batch(
    candidates: list[CandidateInput], rules: dict[str, RuleConfig]
) -> list[CandidateEligibilityResult]:
    return [evaluate_candidate(c, rules) for c in candidates]
