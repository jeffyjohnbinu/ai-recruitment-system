"""
Automated test suite for the Candidate Score Generator (Day 13 deliverable
#3): single/batch generation, manifest loading (inline + file-path data),
persistence via ResultStore, and leaderboard ranking.

Run with:
    python -m pytest ats_scoring_engine/tests/test_generator.py -v
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ats_scoring_engine.generator import (  # noqa: E402
    CandidateScoreGenerator,
    CandidateScoreRequest,
    load_manifest,
)
from ats_scoring_engine.scoring_engine import ATSScoringEngine  # noqa: E402
from ats_scoring_engine.storage import ResultStore  # noqa: E402

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MANIFEST_PATH = FIXTURES_DIR / "manifest_sample.json"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------- #
# generate_one / generate_batch
# --------------------------------------------------------------------- #
def test_generate_one_returns_scored_record():
    generator = CandidateScoreGenerator()
    request = CandidateScoreRequest(
        candidate_id="C001",
        job_id="J001",
        score_overrides={
            "skill_match": 0.9,
            "experience_relevance": 0.8,
            "education_alignment": 0.7,
            "semantic_similarity": 0.6,
        },
    )
    record = generator.generate_one(request)
    assert record.status == "scored"
    assert record.candidate_id == "C001"
    assert 0.0 <= record.final_score <= 1.0


def test_generate_batch_preserves_order_and_scores_all():
    generator = CandidateScoreGenerator()
    requests = [
        CandidateScoreRequest(
            candidate_id=f"C00{i}",
            job_id="J001",
            score_overrides={
                "skill_match": 0.5,
                "experience_relevance": 0.5,
                "education_alignment": 0.5,
                "semantic_similarity": 0.5,
            },
        )
        for i in range(1, 4)
    ]
    records = generator.generate_batch(requests)
    assert [r.candidate_id for r in records] == ["C001", "C002", "C003"]
    assert all(r.status == "scored" for r in records)


def test_generate_one_persists_when_store_provided(tmp_path):
    store = ResultStore(tmp_path)
    generator = CandidateScoreGenerator(store=store)
    request = CandidateScoreRequest(
        candidate_id="C001",
        job_id="J001",
        score_overrides={
            "skill_match": 0.9,
            "experience_relevance": 0.8,
            "education_alignment": 0.7,
            "semantic_similarity": 0.6,
        },
    )
    generator.generate_one(request)

    saved_files = list(tmp_path.glob("*.json"))
    assert len(saved_files) == 1
    saved = json.loads(saved_files[0].read_text(encoding="utf-8"))
    assert saved["candidate_id"] == "C001"


def test_generate_one_does_not_persist_without_store():
    generator = CandidateScoreGenerator()
    request = CandidateScoreRequest(candidate_id="C001", job_id="J001")
    # Should not raise even though no store was configured, and no data given.
    record = generator.generate_one(request)
    assert record.candidate_id == "C001"


# --------------------------------------------------------------------- #
# Manifest loading
# --------------------------------------------------------------------- #
def test_load_manifest_resolves_relative_json_paths():
    requests = load_manifest(MANIFEST_PATH)
    assert len(requests) == 3
    c001 = next(r for r in requests if r.candidate_id == "C001")
    assert c001.job_id == "J001"
    assert c001.role == "software_engineer"
    assert c001.skill_data is not None
    assert c001.experience_data is not None
    assert c001.education_data is not None
    assert c001.semantic_data is not None
    assert c001.skill_data["match_percentage"] == pytest.approx(83.3)


def test_load_manifest_handles_partial_data_entry():
    requests = load_manifest(MANIFEST_PATH)
    c003 = next(r for r in requests if r.candidate_id == "C003")
    assert c003.skill_data is not None
    assert c003.experience_data is None
    assert c003.education_data is None
    assert c003.semantic_data is None


def test_load_manifest_supports_inline_data(tmp_path):
    manifest = [
        {
            "candidate_id": "C900",
            "job_id": "J900",
            "role": "sales",
            "skill_data": {"matched_skills": ["a"], "required_skills": ["a", "b"]},
        }
    ]
    manifest_path = tmp_path / "inline_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    requests = load_manifest(manifest_path)
    assert len(requests) == 1
    assert requests[0].candidate_id == "C900"
    assert requests[0].role == "sales"
    assert requests[0].skill_data == {"matched_skills": ["a"], "required_skills": ["a", "b"]}
    assert requests[0].experience_data is None


def test_load_manifest_missing_file_path_yields_none(tmp_path):
    manifest = [
        {
            "candidate_id": "C901",
            "job_id": "J901",
            "skill_json": "does_not_exist.json",
        }
    ]
    manifest_path = tmp_path / "broken_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    requests = load_manifest(manifest_path)
    assert requests[0].skill_data is None


# --------------------------------------------------------------------- #
# End-to-end batch generation from manifest
# --------------------------------------------------------------------- #
def test_generate_from_manifest_scores_all_candidates(tmp_path):
    store = ResultStore(tmp_path)
    generator = CandidateScoreGenerator(store=store)
    records = generator.generate_from_manifest(MANIFEST_PATH)

    assert len(records) == 3
    by_id = {r.candidate_id: r for r in records}
    assert by_id["C001"].status == "scored"
    assert by_id["C002"].status == "scored"
    # C003 has only skill data; default engine (renormalize) still scores it.
    assert by_id["C003"].status == "scored"
    assert by_id["C003"].components_missing == [
        "experience_relevance",
        "education_alignment",
        "semantic_similarity",
    ]

    # Persistence happened for all three.
    assert len(list(tmp_path.glob("*.json"))) == 3


def test_generate_from_manifest_strong_candidate_outranks_weak_candidate():
    generator = CandidateScoreGenerator()
    records = generator.generate_from_manifest(MANIFEST_PATH)
    by_id = {r.candidate_id: r for r in records}
    assert by_id["C001"].final_score > by_id["C002"].final_score


# --------------------------------------------------------------------- #
# Leaderboard
# --------------------------------------------------------------------- #
def test_leaderboard_orders_best_first():
    generator = CandidateScoreGenerator()
    records = generator.generate_from_manifest(MANIFEST_PATH)
    leaderboard = generator.build_leaderboard(records)

    assert leaderboard[0]["candidate_id"] == "C001"
    scores = [entry["final_score"] for entry in leaderboard]
    assert scores == sorted(scores, reverse=True)
    assert [entry["rank"] for entry in leaderboard] == [1, 2, 3]


def test_leaderboard_sorts_insufficient_data_last():
    strict_engine = ATSScoringEngine(missing_data_mode="strict", max_missing_for_strict=0)
    generator = CandidateScoreGenerator(engine=strict_engine)

    requests = [
        CandidateScoreRequest(
            candidate_id="STRONG",
            job_id="J001",
            score_overrides={
                "skill_match": 0.9,
                "experience_relevance": 0.9,
                "education_alignment": 0.9,
                "semantic_similarity": 0.9,
            },
        ),
        CandidateScoreRequest(candidate_id="NODATA", job_id="J001"),  # everything missing
    ]
    leaderboard = generator.generate_leaderboard(requests)

    assert leaderboard[0]["candidate_id"] == "STRONG"
    assert leaderboard[1]["candidate_id"] == "NODATA"
    assert leaderboard[1]["status"] == "insufficient_data"
    assert leaderboard[1]["final_score"] is None


def test_generate_leaderboard_convenience_method_matches_manual_flow():
    generator = CandidateScoreGenerator()
    requests = load_manifest(MANIFEST_PATH)

    via_convenience = generator.generate_leaderboard(requests)
    via_manual = generator.build_leaderboard(generator.generate_batch(requests))

    assert via_convenience == via_manual


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
