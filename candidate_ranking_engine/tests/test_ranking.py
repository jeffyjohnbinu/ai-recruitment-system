"""
Automated test suite for the Candidate Ranking & Shortlisting Engine.

Run with:
    python -m pytest candidate_ranking_engine/tests/test_ranking.py -v

Or use tests/run_tests.py to also generate a timestamped log file
(deliverable: "Test result logs").
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from candidate_ranking_engine.loader import (  # noqa: E402
    MatchRecordError,
    load_from_directory,
    load_from_file,
    parse_match_record,
)
from candidate_ranking_engine.ranker import (  # noqa: E402
    ZONE_AUTO_REJECT,
    ZONE_REVIEW,
    ZONE_SHORTLIST,
    CandidateRankingEngine,
    InvalidConfigError,
    NoCandidatesError,
    RankingConfig,
)
from candidate_ranking_engine.storage import CandidateMatchInput  # noqa: E402

SAMPLES_DIR = Path(__file__).parent / "sample_data"
OUTPUT_DIR = Path(__file__).parent / "test_outputs"


@pytest.fixture(scope="module")
def engine():
    return CandidateRankingEngine(output_dir=OUTPUT_DIR)


def _make(candidate_id, score, skills=None, job_id="job_100", name=None):
    return CandidateMatchInput(
        candidate_id=candidate_id,
        job_id=job_id,
        overall_score=score,
        candidate_name=name,
        section_scores={"skills": skills} if skills is not None else {},
    )


# --------------------------------------------------------------------- #
# Loader — tolerant parsing of Day 12 (and alias-key) records
# --------------------------------------------------------------------- #
def test_loader_parses_standard_record():
    record = load_from_file(SAMPLES_DIR / "cand_001_job_100.json")
    assert record.candidate_id == "cand_001"
    assert record.job_id == "job_100"
    assert record.overall_score == pytest.approx(0.91)
    assert record.band == "Strong Match"
    assert record.section_scores["skills"] == pytest.approx(0.95)


def test_loader_parses_alternate_key_names():
    record = load_from_file(SAMPLES_DIR / "cand_006_job_100_altkeys.json")
    assert record.candidate_id == "cand_006"
    assert record.job_id == "job_100"
    assert record.overall_score == pytest.approx(0.80)
    assert record.candidate_name == "Legacy Format Candidate"
    assert record.section_scores["skills"] == pytest.approx(0.82)


def test_loader_rejects_missing_required_fields():
    with pytest.raises(MatchRecordError):
        parse_match_record({"note": "missing everything"})


def test_loader_directory_skips_malformed_and_filters_by_job(caplog):
    records = load_from_directory(SAMPLES_DIR, job_id="job_100")
    ids = {r.candidate_id for r in records}
    # job_200 candidate excluded, malformed record skipped
    assert "cand_005" not in ids
    assert ids == {"cand_001", "cand_002", "cand_003", "cand_004", "cand_006"}


def test_loader_directory_no_job_filter_returns_all_parseable():
    records = load_from_directory(SAMPLES_DIR)
    ids = {r.candidate_id for r in records}
    assert "cand_005" in ids  # job_200 candidate included when no filter given


# --------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------- #
def test_config_rejects_inverted_thresholds():
    with pytest.raises(InvalidConfigError):
        RankingConfig(shortlist_threshold=0.4, review_threshold=0.6)


def test_config_rejects_invalid_top_n():
    with pytest.raises(InvalidConfigError):
        RankingConfig(top_n=0)


def test_config_defaults_are_valid():
    config = RankingConfig()
    assert 0.0 <= config.review_threshold <= config.shortlist_threshold <= 1.0


# --------------------------------------------------------------------- #
# Sorting / ranking
# --------------------------------------------------------------------- #
def test_ranking_sorts_by_score_descending(engine):
    candidates = [_make("a", 0.5), _make("b", 0.9), _make("c", 0.7)]
    report = engine.rank_job(candidates, job_id="job_100")
    scores = [c.overall_score for c in report.ranked_candidates]
    assert scores == sorted(scores, reverse=True)
    assert report.ranked_candidates[0].candidate_id == "b"


def test_ranking_assigns_sequential_ranks(engine):
    candidates = [_make("a", 0.5), _make("b", 0.9), _make("c", 0.7)]
    report = engine.rank_job(candidates, job_id="job_100")
    assert [c.rank for c in report.ranked_candidates] == [1, 2, 3]


def test_ranking_tiebreaks_on_skills_score_then_id(engine):
    # Both score 0.62; cand_004 has lower skills (0.50) than cand_002 (0.60)
    # in the fixture files -- but here we build tied cases explicitly.
    candidates = [
        _make("z_low_skill", 0.62, skills=0.40),
        _make("a_high_skill", 0.62, skills=0.90),
        _make("m_mid_skill", 0.62, skills=0.60),
    ]
    report = engine.rank_job(candidates, job_id="job_100")
    ordered_ids = [c.candidate_id for c in report.ranked_candidates]
    assert ordered_ids == ["a_high_skill", "m_mid_skill", "z_low_skill"]


def test_ranking_is_deterministic_across_runs(engine):
    candidates = [_make("a", 0.5), _make("b", 0.9), _make("c", 0.7)]
    report1 = engine.rank_job(list(candidates), job_id="job_100")
    report2 = engine.rank_job(list(candidates), job_id="job_100")
    ids1 = [c.candidate_id for c in report1.ranked_candidates]
    ids2 = [c.candidate_id for c in report2.ranked_candidates]
    assert ids1 == ids2


# --------------------------------------------------------------------- #
# Zoning / shortlisting thresholds
# --------------------------------------------------------------------- #
def test_zone_boundaries_are_inclusive_at_threshold():
    engine = CandidateRankingEngine(
        config=RankingConfig(shortlist_threshold=0.75, review_threshold=0.50),
        output_dir=OUTPUT_DIR,
    )
    assert engine.classify_zone(0.75) == ZONE_SHORTLIST
    assert engine.classify_zone(0.749999) == ZONE_REVIEW
    assert engine.classify_zone(0.50) == ZONE_REVIEW
    assert engine.classify_zone(0.4999) == ZONE_AUTO_REJECT
    assert engine.classify_zone(1.0) == ZONE_SHORTLIST
    assert engine.classify_zone(0.0) == ZONE_AUTO_REJECT


def test_zone_counts_match_candidate_assignments(engine):
    candidates = [
        _make("hi", 0.90),  # shortlist
        _make("mid", 0.60),  # review
        _make("lo", 0.20),  # auto-reject
    ]
    report = engine.rank_job(candidates, job_id="job_100")
    assert report.shortlist_count == 1
    assert report.review_count == 1
    assert report.auto_reject_count == 1
    zones = {c.candidate_id: c.zone for c in report.ranked_candidates}
    assert zones["hi"] == ZONE_SHORTLIST
    assert zones["mid"] == ZONE_REVIEW
    assert zones["lo"] == ZONE_AUTO_REJECT


def test_custom_thresholds_change_zoning():
    strict_engine = CandidateRankingEngine(
        config=RankingConfig(shortlist_threshold=0.95, review_threshold=0.85),
        output_dir=OUTPUT_DIR,
    )
    candidates = [_make("a", 0.90)]
    report = strict_engine.rank_job(candidates, job_id="job_100")
    assert report.ranked_candidates[0].zone == ZONE_REVIEW  # would be shortlist under defaults


# --------------------------------------------------------------------- #
# Top-N / recruiter output
# --------------------------------------------------------------------- #
def test_top_n_respects_config(engine_local=None):
    engine = CandidateRankingEngine(config=RankingConfig(top_n=2), output_dir=OUTPUT_DIR)
    candidates = [_make(str(i), score) for i, score in enumerate([0.9, 0.1, 0.5, 0.7, 0.3])]
    report = engine.rank_job(candidates, job_id="job_100")
    assert len(report.top_candidates) == 2
    assert report.top_candidates[0].overall_score == pytest.approx(0.9)
    assert report.top_candidates[1].overall_score == pytest.approx(0.7)


def test_top_n_capped_when_fewer_candidates_than_n(engine):
    candidates = [_make("only_one", 0.6)]
    report = engine.rank_job(candidates, job_id="job_100")
    assert len(report.top_candidates) == 1


def test_reason_strings_reflect_zone(engine):
    candidates = [_make("hi", 0.9), _make("mid", 0.6), _make("lo", 0.1)]
    report = engine.rank_job(candidates, job_id="job_100")
    reasons = {c.candidate_id: c.reason for c in report.ranked_candidates}
    assert "shortlist" in reasons["hi"].lower()
    assert "review" in reasons["mid"].lower()
    assert "below" in reasons["lo"].lower()


# --------------------------------------------------------------------- #
# Metadata envelope
# --------------------------------------------------------------------- #
def test_ranked_candidates_carry_metadata_envelope(engine):
    candidates = [_make("a", 0.9)]
    report = engine.rank_job(candidates, job_id="job_100")
    c = report.ranked_candidates[0]
    assert c.schema_version
    assert c.model_version
    assert c.pipeline_version
    assert c.generated_at


def test_report_carries_job_and_threshold_context(engine):
    candidates = [_make("a", 0.9)]
    report = engine.rank_job(candidates, job_id="job_100")
    assert report.job_id == "job_100"
    assert report.shortlist_threshold == engine.config.shortlist_threshold
    assert report.review_threshold == engine.config.review_threshold


# --------------------------------------------------------------------- #
# Mismatched job_id filtering within rank_job itself
# --------------------------------------------------------------------- #
def test_rank_job_excludes_mismatched_job_ids(engine, caplog):
    candidates = [
        _make("a", 0.9, job_id="job_100"),
        _make("b", 0.8, job_id="job_999"),
    ]
    report = engine.rank_job(candidates, job_id="job_100")
    assert report.total_candidates == 1
    assert report.ranked_candidates[0].candidate_id == "a"


# --------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------- #
def test_rank_job_raises_on_empty_input(engine):
    with pytest.raises(NoCandidatesError):
        engine.rank_job([], job_id="job_100")


def test_rank_job_raises_when_all_candidates_mismatch_job(engine):
    candidates = [_make("a", 0.9, job_id="job_other")]
    with pytest.raises(NoCandidatesError):
        engine.rank_job(candidates, job_id="job_100")


# --------------------------------------------------------------------- #
# End-to-end: directory -> ranking -> persisted output
# --------------------------------------------------------------------- #
def test_rank_directory_end_to_end_and_persists_output(engine):
    report = engine.rank_directory(SAMPLES_DIR, job_id="job_100")
    assert report.total_candidates == 5  # cand_001,002,003,004,006 (job_100 only)
    assert report.job_id == "job_100"

    json_path = OUTPUT_DIR / "structured" / "job_100_ranking.json"
    csv_path = OUTPUT_DIR / "recruiter_csv" / "job_100_ranking.csv"
    assert json_path.exists()
    assert csv_path.exists()

    csv_text = csv_path.read_text(encoding="utf-8")
    assert "candidate_id" in csv_text.splitlines()[0]
    assert "cand_001" in csv_text


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
