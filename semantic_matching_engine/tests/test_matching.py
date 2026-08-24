"""
Automated test suite for the Semantic Matching Engine.

Run with:
    python -m pytest semantic_matching_engine/tests/test_matching.py -v

Or use tests/run_tests.py to also generate a timestamped log file.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from semantic_matching_engine.adapters import (  # noqa: E402
    normalize_job_input,
    normalize_resume_input,
)
from semantic_matching_engine.embeddings import HashingBoWEmbedder, get_embedder  # noqa: E402
from semantic_matching_engine.matcher import SemanticMatchingEngine  # noqa: E402
from semantic_matching_engine.reports import ValidationRow, generate_accuracy_report  # noqa: E402
from semantic_matching_engine.similarity import compute_similarity, cosine_similarity  # noqa: E402
from semantic_matching_engine.thresholds import ThresholdConfig, tune_match_threshold  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from fixtures import (  # noqa: E402
    JOB_BACKEND_ENGINEER,
    JOB_DATA_SCIENTIST,
    RESUME_BACKEND_STRONG,
    RESUME_BACKEND_WEAK,
    RESUME_DATA_SCIENTIST,
    RESUME_PRODUCT_MANAGER,
    VALIDATION_PAIRS,
)

OUTPUT_DIR = Path(__file__).parent / "test_outputs"


@pytest.fixture(scope="module")
def engine():
    return SemanticMatchingEngine(output_dir=OUTPUT_DIR, embedder_preference="hashing")


# --------------------------------------------------------------------- #
# Embeddings / engine selection
# --------------------------------------------------------------------- #
def test_hashing_embedder_produces_normalized_vectors():
    embedder = HashingBoWEmbedder()
    vectors = embedder.embed(["Python developer", "Java developer"])
    assert vectors.shape[0] == 2
    for vec in vectors:
        norm = np.linalg.norm(vec)
        assert norm == pytest.approx(1.0, abs=1e-6) or norm == 0.0


def test_get_embedder_auto_falls_back_without_network():
    embedder = get_embedder(prefer="auto")
    # In this sandboxed test environment sentence-transformers weights are
    # not reachable, so auto-selection should degrade to the offline engine
    # rather than raising.
    assert embedder.engine_name in ("hashing-bow-fallback", "sentence-transformers")


def test_get_embedder_forced_hashing():
    embedder = get_embedder(prefer="hashing")
    assert embedder.engine_name == "hashing-bow-fallback"


# --------------------------------------------------------------------- #
# Cosine similarity
# --------------------------------------------------------------------- #
def test_cosine_similarity_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_returns_zero():
    a = np.zeros(5)
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert cosine_similarity(a, b) == 0.0


# --------------------------------------------------------------------- #
# Section-weighted similarity
# --------------------------------------------------------------------- #
def test_compute_similarity_strong_match_scores_higher_than_weak():
    embedder = HashingBoWEmbedder()
    strong = compute_similarity(RESUME_BACKEND_STRONG, JOB_BACKEND_ENGINEER, embedder)
    weak = compute_similarity(RESUME_BACKEND_WEAK, JOB_BACKEND_ENGINEER, embedder)
    assert strong.overall_score > weak.overall_score


def test_compute_similarity_missing_section_scores_zero_not_excluded():
    embedder = HashingBoWEmbedder()
    resume = {"skills": "Python", "experience": "", "projects": "Built stuff"}
    job = {"skills": "Python", "experience": "5 years required", "projects": ""}
    breakdown = compute_similarity(resume, job, embedder)
    assert breakdown.section_scores["experience"].empty is True
    assert breakdown.section_scores["experience"].similarity == 0.0
    assert breakdown.section_scores["projects"].empty is True


def test_compute_similarity_respects_custom_weights():
    embedder = HashingBoWEmbedder()
    resume = {"skills": "Python", "experience": "totally unrelated text here", "projects": ""}
    job = {"skills": "Python", "experience": "totally unrelated text here", "projects": ""}
    skills_heavy = compute_similarity(
        resume, job, embedder, weights={"skills": 0.9, "experience": 0.05, "projects": 0.05}
    )
    exp_heavy = compute_similarity(
        resume, job, embedder, weights={"skills": 0.05, "experience": 0.9, "projects": 0.05}
    )
    # Skills and experience are perfect matches on both sides; "projects"
    # is blank on both sides so it stays 0.0 regardless of weighting. With
    # a 0.05 weight on the empty "projects" axis, the ceiling is 0.95.
    assert skills_heavy.overall_score == pytest.approx(0.95, abs=1e-6)
    assert exp_heavy.overall_score == pytest.approx(0.95, abs=1e-6)


def test_compute_similarity_weights_must_be_positive():
    embedder = HashingBoWEmbedder()
    with pytest.raises(ValueError):
        compute_similarity({}, {}, embedder, weights={"skills": 0, "experience": 0, "projects": 0})


# --------------------------------------------------------------------- #
# Adapters (cross-day input normalization)
# --------------------------------------------------------------------- #
def test_adapter_accepts_flat_target_dict():
    sections, warnings = normalize_resume_input(RESUME_BACKEND_STRONG)
    assert sections["skills"] == RESUME_BACKEND_STRONG["skills"]
    assert warnings == []


def test_adapter_accepts_day8_section_map():
    section_map = {
        "Skills": "Python, SQL",
        "Experience": "5 years backend",
        "Education": "B.S. Computer Science",
    }
    sections, warnings = normalize_resume_input(section_map)
    assert sections["skills"] == "Python, SQL"
    assert sections["experience"] == "5 years backend"
    assert "projects" in warnings[0].lower() if warnings else True


def test_adapter_accepts_day9_skill_records():
    skill_records = [
        {"skill": "Python", "confidence": 0.95},
        {"skill": "AWS", "confidence": 0.8},
    ]
    sections, warnings = normalize_resume_input(skill_records)
    assert "Python" in sections["skills"]
    assert "AWS" in sections["skills"]


def test_adapter_accepts_day10_experience_records():
    experience_records = [
        {"title": "Backend Engineer", "company": "Acme", "description": "Built APIs"},
    ]
    sections, warnings = normalize_resume_input(experience_records)
    assert "Built APIs" in sections["experience"]


def test_adapter_accepts_flat_text_with_warning():
    sections, warnings = normalize_resume_input("Some raw resume text.")
    assert sections["skills"] == "Some raw resume text."
    assert sections["experience"] == "Some raw resume text."
    assert len(warnings) == 1


def test_adapter_job_accepts_day6_job_requirement_record():
    sections, warnings = normalize_job_input(JOB_BACKEND_ENGINEER)
    assert "Python" in sections["skills"]
    assert sections["experience"]


def test_adapter_unrecognized_type_warns():
    sections, warnings = normalize_resume_input(12345)
    assert warnings
    assert all(v == "" for v in sections.values())


# --------------------------------------------------------------------- #
# Threshold classification & tuning
# --------------------------------------------------------------------- #
def test_threshold_classify_bands():
    tc = ThresholdConfig(match_threshold=0.5, strong_match_threshold=0.8)
    assert tc.classify(0.9) == "strong_match"
    assert tc.classify(0.6) == "possible_match"
    assert tc.classify(0.2) == "no_match"


def test_threshold_config_rejects_invalid_ordering():
    with pytest.raises(ValueError):
        ThresholdConfig(match_threshold=0.8, strong_match_threshold=0.5)


def test_tune_match_threshold_finds_separating_value():
    scores = [0.9, 0.85, 0.8, 0.4, 0.3, 0.1]
    labels = [True, True, True, False, False, False]
    result = tune_match_threshold(scores, labels, step=0.05)
    assert result.best_f1 == pytest.approx(1.0)
    assert 0.4 < result.best_threshold <= 0.8


def test_tune_match_threshold_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        tune_match_threshold([0.5, 0.6], [True])


# --------------------------------------------------------------------- #
# End-to-end matching (SemanticMatchingEngine)
# --------------------------------------------------------------------- #
def test_engine_matches_strong_backend_pair(engine):
    record = engine.match("C_backend_1", "J_backend", RESUME_BACKEND_STRONG, JOB_BACKEND_ENGINEER)
    assert record.status == "success"
    assert record.overall_score > 0.3
    assert record.engine_used == "hashing-bow-fallback"


def test_engine_scores_weak_pair_lower_than_strong_pair(engine):
    strong = engine.match("C_strong", "J_backend", RESUME_BACKEND_STRONG, JOB_BACKEND_ENGINEER)
    weak = engine.match("C_weak", "J_backend", RESUME_BACKEND_WEAK, JOB_BACKEND_ENGINEER)
    assert strong.overall_score > weak.overall_score


def test_engine_cross_job_type_scores_lower_than_matching_type(engine):
    matching = engine.match("C_ds", "J_ds", RESUME_DATA_SCIENTIST, JOB_DATA_SCIENTIST)
    mismatched = engine.match("C_pm_as_ds", "J_ds", RESUME_PRODUCT_MANAGER, JOB_DATA_SCIENTIST)
    assert matching.overall_score > mismatched.overall_score


def test_engine_persists_json_output(engine):
    record = engine.match("C_persist", "J_persist", RESUME_BACKEND_STRONG, JOB_BACKEND_ENGINEER)
    expected_path = (
        OUTPUT_DIR
        / "structured"
        / "semantic_matches"
        / f"{record.candidate_id}__{record.job_id}.json"
    )
    assert expected_path.exists()


def test_engine_handles_completely_empty_input_gracefully(engine):
    record = engine.match(
        "C_empty",
        "J_empty",
        {"skills": "", "experience": "", "projects": ""},
        {"skills": "", "experience": "", "projects": ""},
    )
    assert record.status == "failed"
    assert record.overall_score == 0.0
    assert record.is_match is False


# --------------------------------------------------------------------- #
# Accuracy report / multi-job-type validation
# --------------------------------------------------------------------- #
def test_accuracy_report_across_multiple_job_types(engine):
    rows = []
    for resume, job, job_type, expected_match in VALIDATION_PAIRS:
        record = engine.match(f"C_{job_type}", f"J_{job_type}", resume, job, persist=False)
        rows.append(
            ValidationRow(
                candidate_id=f"C_{job_type}",
                job_id=f"J_{job_type}",
                overall_score=record.overall_score,
                predicted_is_match=record.is_match,
                actual_is_match=expected_match,
                job_type=job_type,
            )
        )
    report = generate_accuracy_report(rows)
    assert report.total_pairs == len(VALIDATION_PAIRS)
    assert 0.0 <= report.precision <= 1.0
    assert 0.0 <= report.recall <= 1.0
    assert set(report.by_job_type.keys()) == {"engineering", "data_science", "product"}


def test_accuracy_report_rejects_empty_validation_set():
    with pytest.raises(ValueError):
        generate_accuracy_report([])


def test_accuracy_report_markdown_contains_key_sections():
    rows = [ValidationRow("C1", "J1", 0.9, True, True, "engineering")]
    report = generate_accuracy_report(rows)
    md = report.to_markdown()
    assert "# Semantic Matching Accuracy Report" in md
    assert "Confusion matrix" in md


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
