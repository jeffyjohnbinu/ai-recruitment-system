"""
Automated test suite for the Day 17 ATS System Testing module.

Run with:
    python -m pytest ats_system_testing/tests/test_ats_system_testing.py -v

Or use tests/run_tests.py to also generate a timestamped log file
(deliverable: "Test result logs").

Two layers are tested:
  1. The Day 17 module's own machinery (harness, metrics math, backlog
     rules) using small synthetic CaseResult fixtures -- this must pass
     regardless of which upstream engines are installed.
  2. An end-to-end smoke run of the full fixture set through whatever
     pipeline is actually resolvable in this environment (real engines
     if present, fallback scorer otherwise), asserting the harness
     produces well-formed output for every case.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ats_system_testing.backlog import build_backlog  # noqa: E402
from ats_system_testing.fixtures import ATSTestCase, GroundTruth, get_test_cases  # noqa: E402
from ats_system_testing.harness import ATSTestHarness, CaseResult  # noqa: E402
from ats_system_testing.metrics import compute_metrics  # noqa: E402
from ats_system_testing.pipeline_adapter import PipelineOutput, run_ats_pipeline  # noqa: E402


def _make_case(
    case_id, shortlisted_gt, recommendation_gt="advance", role_type="tech", seniority="senior"
):
    return ATSTestCase(
        case_id=case_id,
        role_type=role_type,
        seniority=seniority,
        expected_outcome="match" if shortlisted_gt else "mismatch",
        candidate_name="Test Candidate",
        resume_text="irrelevant for synthetic unit tests",
        job_description="irrelevant for synthetic unit tests",
        ground_truth=GroundTruth(
            shortlisted=shortlisted_gt,
            recommendation=recommendation_gt,
            expected_matched_skills=[],
            reviewer_notes="synthetic fixture for unit testing",
        ),
    )


def _make_result(case, ai_shortlisted, ai_recommendation="advance"):
    return CaseResult(
        test_case=case,
        pipeline_output=PipelineOutput(
            score=0.9 if ai_shortlisted else 0.1,
            shortlisted=ai_shortlisted,
            recommendation=ai_recommendation,
            matched_keywords=[],
            engine_used="synthetic",
        ),
    )


# --------------------------------------------------------------------- #
# Fixture dataset sanity
# --------------------------------------------------------------------- #
def test_fixture_set_covers_all_four_segments():
    cases = get_test_cases()
    segments = {(c.role_type, c.seniority) for c in cases}
    assert segments == {
        ("tech", "senior"),
        ("tech", "fresher"),
        ("non_tech", "senior"),
        ("non_tech", "fresher"),
    }


def test_fixture_set_has_both_match_and_mismatch_per_segment():
    cases = get_test_cases()
    for role_type in ("tech", "non_tech"):
        for seniority in ("senior", "fresher"):
            segment = [c for c in cases if c.role_type == role_type and c.seniority == seniority]
            outcomes = {c.expected_outcome for c in segment}
            assert outcomes == {"match", "mismatch"}, f"{role_type}/{seniority} missing a case type"


def test_get_test_cases_filters_by_role_type():
    tech_cases = get_test_cases(role_type="tech")
    assert tech_cases
    assert all(c.role_type == "tech" for c in tech_cases)


def test_get_test_cases_filters_by_seniority():
    fresher_cases = get_test_cases(seniority="fresher")
    assert fresher_cases
    assert all(c.seniority == "fresher" for c in fresher_cases)


def test_case_ids_are_unique():
    cases = get_test_cases()
    ids = [c.case_id for c in cases]
    assert len(ids) == len(set(ids))


# --------------------------------------------------------------------- #
# Pipeline adapter (fallback path -- must always work with zero deps)
# --------------------------------------------------------------------- #
def test_pipeline_adapter_returns_output_for_any_pair():
    output = run_ats_pipeline(
        "Python developer with AWS experience", "Looking for a Python developer"
    )
    assert isinstance(output, PipelineOutput)
    assert 0.0 <= output.score <= 1.0
    assert output.recommendation in ("advance", "hold", "reject")


def test_pipeline_adapter_handles_empty_job_description():
    output = run_ats_pipeline("Python developer", "")
    assert output.score == 0.0
    assert output.shortlisted is False


def test_pipeline_adapter_records_which_engine_was_used():
    output = run_ats_pipeline("Python developer", "Python developer needed")
    assert output.engine_used  # non-empty string, whichever tier resolved


# --------------------------------------------------------------------- #
# Harness end-to-end smoke test
# --------------------------------------------------------------------- #
def test_harness_runs_full_fixture_set_without_error():
    harness = ATSTestHarness()
    results = harness.run()
    assert len(results) == len(get_test_cases())
    for r in results:
        assert isinstance(r.pipeline_output, PipelineOutput)
        assert isinstance(r.is_correct, bool)


def test_harness_respects_custom_case_subset():
    subset = get_test_cases(role_type="tech")
    harness = ATSTestHarness(test_cases=subset)
    results = harness.run()
    assert len(results) == len(subset)


# --------------------------------------------------------------------- #
# Metrics math (synthetic, deterministic -- independent of which real
# engine is installed)
# --------------------------------------------------------------------- #
def test_metrics_perfect_agreement_gives_full_scores():
    cases = [_make_case(f"C{i}", shortlisted_gt=(i % 2 == 0)) for i in range(4)]
    results = [_make_result(c, ai_shortlisted=c.ground_truth.shortlisted) for c in cases]
    metrics = compute_metrics(results)
    assert metrics.overall.accuracy == 1.0
    assert metrics.overall.precision == 1.0
    assert metrics.overall.recall == 1.0
    assert metrics.overall.f1 == 1.0
    assert metrics.mismatches == []


def test_metrics_detects_false_positive():
    case = _make_case("FP-1", shortlisted_gt=False)
    result = _make_result(case, ai_shortlisted=True)
    metrics = compute_metrics([result])
    assert metrics.overall.false_positive == 1
    assert metrics.mismatches[0].mismatch_type == "false_positive"


def test_metrics_detects_false_negative():
    case = _make_case("FN-1", shortlisted_gt=True)
    result = _make_result(case, ai_shortlisted=False)
    metrics = compute_metrics([result])
    assert metrics.overall.false_negative == 1
    assert metrics.mismatches[0].mismatch_type == "false_negative"


def test_metrics_precision_recall_known_values():
    # 2 true positives, 1 false positive, 1 false negative, 0 true negatives
    cases_and_ai = [
        (_make_case("TP1", shortlisted_gt=True), True),
        (_make_case("TP2", shortlisted_gt=True), True),
        (_make_case("FP1", shortlisted_gt=False), True),
        (_make_case("FN1", shortlisted_gt=True), False),
    ]
    results = [_make_result(c, ai) for c, ai in cases_and_ai]
    metrics = compute_metrics(results)
    # precision = TP / (TP+FP) = 2/3, recall = TP / (TP+FN) = 2/3
    assert metrics.overall.precision == pytest.approx(2 / 3, rel=1e-3)
    assert metrics.overall.recall == pytest.approx(2 / 3, rel=1e-3)


def test_metrics_segment_breakdown_present():
    cases = [
        _make_case("T1", shortlisted_gt=True, role_type="tech", seniority="senior"),
        _make_case("N1", shortlisted_gt=True, role_type="non_tech", seniority="fresher"),
    ]
    results = [_make_result(c, ai_shortlisted=True) for c in cases]
    metrics = compute_metrics(results)
    assert "tech" in metrics.by_role_type
    assert "non_tech" in metrics.by_role_type
    assert "senior" in metrics.by_seniority
    assert "fresher" in metrics.by_seniority


# --------------------------------------------------------------------- #
# Improvement backlog
# --------------------------------------------------------------------- #
def test_backlog_flags_false_positive_as_p0():
    case = _make_case("FP-1", shortlisted_gt=False)
    result = _make_result(case, ai_shortlisted=True)
    metrics = compute_metrics([result])
    backlog = build_backlog(metrics)
    assert any(item.priority == "P0" and "false-positive" in item.title.lower() for item in backlog)


def test_backlog_flags_false_negative_as_p0():
    case = _make_case("FN-1", shortlisted_gt=True)
    result = _make_result(case, ai_shortlisted=False)
    metrics = compute_metrics([result])
    backlog = build_backlog(metrics)
    assert any(item.priority == "P0" and "false negative" in item.title.lower() for item in backlog)


def test_backlog_non_empty_even_with_perfect_agreement():
    cases = [_make_case(f"C{i}", shortlisted_gt=True) for i in range(2)]
    results = [_make_result(c, ai_shortlisted=True) for c in cases]
    metrics = compute_metrics(results)
    backlog = build_backlog(metrics)
    assert len(backlog) >= 1  # still recommends expanding coverage


def test_backlog_items_reference_real_case_ids():
    case = _make_case("FP-REF", shortlisted_gt=False)
    result = _make_result(case, ai_shortlisted=True)
    metrics = compute_metrics([result])
    backlog = build_backlog(metrics)
    fp_items = [i for i in backlog if "false-positive" in i.title.lower()]
    assert fp_items
    assert "FP-REF" in fp_items[0].affected_cases


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
