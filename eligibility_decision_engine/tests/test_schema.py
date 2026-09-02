from eligibility_decision_engine.schema import (
    CandidateEligibilityResult,
    CandidateInput,
    EligibilityStatus,
    RuleConfig,
    RuleReason,
)


def test_candidate_input_to_dict_excludes_raw():
    c = CandidateInput(candidate_id="C1", job_role="x", ats_score=80, raw={"foo": "bar"})
    d = c.to_dict()
    assert "raw" not in d
    assert d["candidate_id"] == "C1"


def test_rule_config_defaults():
    r = RuleConfig(job_role="x")
    assert r.min_ats_score == 0.0
    assert r.mandatory_skills == []
    assert r.allow_remote is True
    assert r.max_experience_years is None


def test_result_to_dict_serializes_status_and_reasons():
    res = CandidateEligibilityResult(
        candidate_id="C1",
        job_role="x",
        status=EligibilityStatus.ELIGIBLE,
        ats_score=90,
        reasons=[RuleReason("ats_score", True, "90 >= 75")],
    )
    d = res.to_dict()
    assert d["status"] == "eligible"
    assert d["reasons"][0]["rule"] == "ats_score"
