"""
End-to-end integration tests for the Day 20 ATS Production System.

Unlike the Day 17 test harness (which, per the Day 20 review, silently
fell through to a keyword-overlap fallback due to incorrect import
paths), these tests assert that the REAL Day 5-18 engines are exercised
-- not just that *some* score comes back.

Run with:
    python -m pytest ats_production_system/tests/test_integration.py -v

Or use tests/run_tests.py to also generate a timestamped log
(deliverable: "Test result logs").
"""

import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ats_production_system.pipeline import ATSPipeline  # noqa: E402

DEMO_RESUMES = REPO_ROOT / "demo_datasets" / "resumes"
DEMO_JOBS = REPO_ROOT / "demo_datasets" / "job_descriptions"
OUTPUT_DIR = Path(__file__).parent / "test_outputs"

ALL_CORE_ENGINES = [
    "resume_extraction",
    "jd_parsing",
    "section_classifier",
    "skill_extraction",
    "experience_parsing",
    "education_extraction",
    "semantic_matching",
    "ats_scoring",
    "candidate_ranking",
    "fairness_bias",
    "performance_optimizer",
]


@pytest.fixture(scope="module")
def pipeline():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    # Force the offline hashing embedder: this sandbox has no network
    # access to Hugging Face, and forcing it keeps the suite fast and
    # deterministic instead of retrying a network call on every run.
    return ATSPipeline(output_dir=str(OUTPUT_DIR), embedder_preference="hashing")


@pytest.fixture(scope="module")
def senior_backend_job(pipeline):
    return pipeline.process_job(DEMO_JOBS / "senior_backend_engineer.txt")


@pytest.fixture(scope="module")
def junior_analyst_job(pipeline):
    return pipeline.process_job(DEMO_JOBS / "junior_data_analyst.txt")


# --------------------------------------------------------------------- #
# Engine availability -- the central Day 20 finding: every real engine
# must actually import and initialize, not silently degrade.
# --------------------------------------------------------------------- #
def test_all_core_engines_available(pipeline):
    status = pipeline.engine_status()
    missing = [name for name in ALL_CORE_ENGINES if not status.get(name)]
    assert not missing, f"Expected all core engines available, missing: {missing}"


# --------------------------------------------------------------------- #
# Job parsing (Day 6)
# --------------------------------------------------------------------- #
def test_job_parsing_extracts_required_skills(senior_backend_job):
    assert senior_backend_job["status"] in ("success", "partial")
    required = [s.lower() for s in senior_backend_job["required_skills"]]
    assert "python" in required
    assert "aws" in required


# --------------------------------------------------------------------- #
# Single-candidate scoring -- proves every component is REAL, not
# silently "unavailable" (the exact failure mode found in Day 17).
# --------------------------------------------------------------------- #
def test_strong_candidate_all_components_available(pipeline, senior_backend_job):
    run = pipeline.process_candidate(
        DEMO_RESUMES / "priya_sharma.docx",
        senior_backend_job,
        candidate_id="priya_sharma",
        job_id="job_senior_backend_test",
    )
    assert run.status == "success"
    available = {c["name"]: c["available"] for c in run.component_breakdown}
    assert available == {
        "skill_match": True,
        "experience_relevance": True,
        "education_alignment": True,
        "semantic_similarity": True,
    }, f"Expected all four scoring components available, got: {available}"


def test_strong_candidate_scores_high(pipeline, senior_backend_job):
    run = pipeline.process_candidate(
        DEMO_RESUMES / "priya_sharma.docx",
        senior_backend_job,
        candidate_id="priya_sharma",
        job_id="job_senior_backend_test",
    )
    assert run.final_score is not None
    assert run.final_score >= 0.5, "Strong-fit candidate should score well above a weak fit"


def test_weak_candidate_scores_lower_than_strong_candidate(pipeline, senior_backend_job):
    strong = pipeline.process_candidate(
        DEMO_RESUMES / "priya_sharma.docx",
        senior_backend_job,
        candidate_id="priya_sharma",
        job_id="job_senior_backend_test2",
    )
    weak = pipeline.process_candidate(
        DEMO_RESUMES / "megan_ho.docx",
        senior_backend_job,
        candidate_id="megan_ho",
        job_id="job_senior_backend_test2",
    )
    assert strong.final_score > weak.final_score


def test_scores_are_bounded_0_to_1(pipeline, senior_backend_job, junior_analyst_job):
    for resume in DEMO_RESUMES.glob("*.docx"):
        for job_id, job_record in [
            ("job_bounds_backend", senior_backend_job),
            ("job_bounds_analyst", junior_analyst_job),
        ]:
            run = pipeline.process_candidate(
                resume, job_record, candidate_id=resume.stem + job_id, job_id=job_id
            )
            if run.final_score is not None:
                assert 0.0 <= run.final_score <= 1.0


def test_component_adapter_fixes_field_name_mismatch(pipeline, senior_backend_job):
    """
    Regression guard for the specific bug this Day 20 review found: the
    Semantic Matching Engine emits `overall_score`, not `similarity_score`
    -- if component_adapters.py regresses, semantic_similarity silently
    becomes unavailable again instead of raising, so we check explicitly.
    """
    run = pipeline.process_candidate(
        DEMO_RESUMES / "priya_sharma.docx",
        senior_backend_job,
        candidate_id="priya_sharma",
        job_id="job_field_mapping_test",
    )
    semantic_component = next(
        c for c in run.component_breakdown if c["name"] == "semantic_similarity"
    )
    assert semantic_component["available"] is True
    assert semantic_component["raw_score"] is not None


# --------------------------------------------------------------------- #
# Batch mode: ranking (Day 14) + fairness auditing (Day 15)
# --------------------------------------------------------------------- #
def test_batch_produces_ranked_zones(pipeline):
    resumes = [
        DEMO_RESUMES / "priya_sharma.docx",
        DEMO_RESUMES / "daniel_okafor.docx",
        DEMO_RESUMES / "megan_ho.docx",
    ]
    result = pipeline.run_batch(
        resumes, DEMO_JOBS / "senior_backend_engineer.txt", job_id="job_batch_test"
    )
    summary = result["summary"]
    assert summary.total_candidates == 3
    zones = {c["candidate_id"]: c["zone"] for c in summary.candidates}
    assert zones["priya_sharma"] is not None
    # Strongest candidate should not land in the weakest zone
    assert zones["priya_sharma"] != "Auto-Reject"


def test_batch_runs_fairness_audit(pipeline):
    resumes = [
        DEMO_RESUMES / "priya_sharma.docx",
        DEMO_RESUMES / "daniel_okafor.docx",
        DEMO_RESUMES / "megan_ho.docx",
    ]
    result = pipeline.run_batch(
        resumes, DEMO_JOBS / "senior_backend_engineer.txt", job_id="job_fairness_test"
    )
    fairness = result["fairness"]
    assert fairness is not None
    assert len(fairness.per_candidate) == 3


def test_batch_ranking_differs_by_job(pipeline, senior_backend_job):
    """A candidate pool should rank differently for different job postings."""
    resumes = list(DEMO_RESUMES.glob("*.docx"))
    backend_result = pipeline.run_batch(
        resumes, DEMO_JOBS / "senior_backend_engineer.txt", job_id="job_diff_backend"
    )
    analyst_result = pipeline.run_batch(
        resumes, DEMO_JOBS / "junior_data_analyst.txt", job_id="job_diff_analyst"
    )
    backend_top = max(backend_result["summary"].candidates, key=lambda c: c["final_score"] or 0)
    analyst_top = max(analyst_result["summary"].candidates, key=lambda c: c["final_score"] or 0)
    assert backend_top["candidate_id"] != analyst_top["candidate_id"]


# --------------------------------------------------------------------- #
# Graceful degradation -- an engine that fails should not crash the run
# --------------------------------------------------------------------- #
def test_missing_engine_degrades_gracefully(senior_backend_job):
    pipeline = ATSPipeline(output_dir=str(OUTPUT_DIR / "degraded"), embedder_preference="hashing")
    pipeline._engines["education_extraction"] = None
    pipeline._availability["education_extraction"] = False

    run = pipeline.process_candidate(
        DEMO_RESUMES / "priya_sharma.docx",
        senior_backend_job,
        candidate_id="priya_sharma",
        job_id="job_degraded_test",
    )
    assert run.status in ("success", "partial")
    assert "education_extraction" in run.stages_failed
    edu_component = next(c for c in run.component_breakdown if c["name"] == "education_alignment")
    assert edu_component["available"] is False


def test_unsupported_resume_extension_does_not_crash_batch(pipeline, senior_backend_job, tmp_path):
    bad_file = tmp_path / "resume.txt"
    bad_file.write_text("not a real resume")
    run = pipeline.process_candidate(
        bad_file, senior_backend_job, candidate_id="bad_ext", job_id="job_bad_ext_test"
    )
    assert run.status == "failed"
    assert run.error


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
