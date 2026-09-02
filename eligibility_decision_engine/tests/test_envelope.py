from eligibility_decision_engine.envelope import build_envelope
from eligibility_decision_engine.schema import CandidateEligibilityResult, EligibilityStatus


def make_result(status):
    return CandidateEligibilityResult(
        candidate_id="C1", job_role="x", status=status, ats_score=80, reasons=[]
    )


def test_envelope_has_metadata_and_data():
    results = [make_result(EligibilityStatus.ELIGIBLE), make_result(EligibilityStatus.REJECTED)]
    env = build_envelope(results, source="unit_test")
    assert "metadata" in env
    assert "data" in env
    assert env["metadata"]["record_count"] == 2
    assert env["metadata"]["source"] == "unit_test"
    assert env["metadata"]["day"] == "Day 21"


def test_envelope_status_counts():
    results = [
        make_result(EligibilityStatus.ELIGIBLE),
        make_result(EligibilityStatus.ELIGIBLE),
        make_result(EligibilityStatus.REVIEW),
        make_result(EligibilityStatus.REJECTED),
    ]
    env = build_envelope(results)
    counts = env["metadata"]["status_counts"]
    assert counts["eligible"] == 2
    assert counts["review"] == 1
    assert counts["rejected"] == 1


def test_envelope_extra_metadata_merges():
    env = build_envelope([], extra_metadata={"batch_id": "abc123"})
    assert env["metadata"]["batch_id"] == "abc123"
