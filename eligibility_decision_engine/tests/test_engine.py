import pytest

from eligibility_decision_engine.engine import evaluate_batch, evaluate_candidate
from eligibility_decision_engine.schema import CandidateInput, EligibilityStatus, RuleConfig


@pytest.fixture
def rule():
    return RuleConfig(
        job_role="backend_engineer",
        min_ats_score=75,
        review_band=10,
        mandatory_skills=["python"],
        min_experience_years=2,
        max_experience_years=8,
        allowed_locations=["Bengaluru", "Chennai"],
        allow_remote=True,
        required_availability=None,
    )


@pytest.fixture
def rules(rule):
    return {rule.job_role: rule}


def test_eligible_candidate(rules):
    c = CandidateInput(
        candidate_id="C1",
        job_role="backend_engineer",
        ats_score=88,
        skills=["python", "django"],
        experience_years=4,
        location="Bengaluru",
        remote_ok=False,
        availability="immediate",
    )
    res = evaluate_candidate(c, rules)
    assert res.status == EligibilityStatus.ELIGIBLE


def test_score_in_review_band(rules):
    c = CandidateInput(
        candidate_id="C2",
        job_role="backend_engineer",
        ats_score=68,  # 75-10=65 floor, 68 in band
        skills=["python"],
        experience_years=3,
        location="Chennai",
        remote_ok=False,
    )
    res = evaluate_candidate(c, rules)
    assert res.status == EligibilityStatus.REVIEW


def test_score_below_review_floor_rejected(rules):
    c = CandidateInput(
        candidate_id="C3",
        job_role="backend_engineer",
        ats_score=50,
        skills=["python"],
        experience_years=3,
        location="Chennai",
        remote_ok=False,
    )
    res = evaluate_candidate(c, rules)
    assert res.status == EligibilityStatus.REJECTED


def test_missing_mandatory_skill_rejected_even_with_high_score(rules):
    c = CandidateInput(
        candidate_id="C4",
        job_role="backend_engineer",
        ats_score=95,
        skills=["java"],
        experience_years=4,
        location="Bengaluru",
        remote_ok=False,
    )
    res = evaluate_candidate(c, rules)
    assert res.status == EligibilityStatus.REJECTED
    assert any(r.rule == "mandatory_skills" and not r.passed for r in res.reasons)


def test_experience_out_of_range_rejected(rules):
    c = CandidateInput(
        candidate_id="C5",
        job_role="backend_engineer",
        ats_score=90,
        skills=["python"],
        experience_years=1,  # below min 2
        location="Bengaluru",
        remote_ok=False,
    )
    res = evaluate_candidate(c, rules)
    assert res.status == EligibilityStatus.REJECTED


def test_remote_candidate_bypasses_location_restriction(rules):
    c = CandidateInput(
        candidate_id="C6",
        job_role="backend_engineer",
        ats_score=90,
        skills=["python"],
        experience_years=3,
        location="Delhi",
        remote_ok=True,
    )
    res = evaluate_candidate(c, rules)
    assert res.status == EligibilityStatus.ELIGIBLE


def test_location_not_allowed_and_not_remote_rejected(rules):
    c = CandidateInput(
        candidate_id="C7",
        job_role="backend_engineer",
        ats_score=90,
        skills=["python"],
        experience_years=3,
        location="Delhi",
        remote_ok=False,
    )
    res = evaluate_candidate(c, rules)
    assert res.status == EligibilityStatus.REJECTED
    assert any(r.rule == "location" and not r.passed for r in res.reasons)


def test_unknown_job_role_returns_review(rules):
    c = CandidateInput(candidate_id="C8", job_role="unknown_role", ats_score=90)
    res = evaluate_candidate(c, rules)
    assert res.status == EligibilityStatus.REVIEW
    assert res.reasons[0].rule == "job_role_config"


def test_availability_mismatch_rejected():
    rule = RuleConfig(job_role="x", min_ats_score=50, required_availability="immediate")
    c = CandidateInput(candidate_id="C9", job_role="x", ats_score=90, availability="2_weeks")
    res = evaluate_candidate(c, {"x": rule})
    assert res.status == EligibilityStatus.REJECTED


def test_evaluate_batch_preserves_order_and_count(rules):
    cands = [
        CandidateInput(
            candidate_id=f"C{i}",
            job_role="backend_engineer",
            ats_score=90,
            skills=["python"],
            experience_years=3,
            location="Bengaluru",
        )
        for i in range(5)
    ]
    results = evaluate_batch(cands, rules)
    assert len(results) == 5
    assert [r.candidate_id for r in results] == [c.candidate_id for c in cands]
