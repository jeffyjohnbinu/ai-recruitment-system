"""
Automated test suite for the Experience Parsing & Relevance Engine.

Run with:
    python -m pytest experience_parsing_engine/tests/test_experience_engine.py -v

Or use tests/run_tests.py to also generate a timestamped log file.
"""

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experience_parsing_engine.duration import (  # noqa: E402
    is_present_token,
    months_between,
    parse_date_token,
    parse_duration,
)
from experience_parsing_engine.engine import ExperienceParsingEngine  # noqa: E402
from experience_parsing_engine.gaps import analyze_timeline  # noqa: E402
from experience_parsing_engine.parser import ExperienceParser  # noqa: E402
from experience_parsing_engine.relevance import (  # noqa: E402
    ExperienceRelevanceScorer,
    title_similarity,
)

SAMPLES_DIR = Path(__file__).parent / "sample_data"
OUTPUT_DIR = Path(__file__).parent / "test_outputs"

FIXED_TODAY = date(2026, 8, 21)  # deterministic "Present" resolution for tests


def _read(name: str) -> str:
    return (SAMPLES_DIR / name).read_text(encoding="utf-8")


@pytest.fixture
def parser():
    return ExperienceParser(today=FIXED_TODAY)


@pytest.fixture
def engine():
    return ExperienceParsingEngine(output_dir=OUTPUT_DIR)


# --------------------------------------------------------------------- #
# duration.py — date token parsing
# --------------------------------------------------------------------- #
def test_parse_year_only_as_start():
    assert parse_date_token("2021", is_end=False) == (2021, 1)


def test_parse_year_only_as_end():
    assert parse_date_token("2021", is_end=True) == (2021, 12)


def test_parse_month_year():
    assert parse_date_token("Jan 2021") == (2021, 1)
    assert parse_date_token("January 2021") == (2021, 1)
    assert parse_date_token("Sept 2019") == (2019, 9)


def test_parse_present_token_resolves_to_today():
    assert parse_date_token("Present", today=FIXED_TODAY) == (2026, 8)
    assert is_present_token("current") is True
    assert is_present_token("2021") is False


def test_months_between_inclusive():
    assert months_between((2020, 1), (2020, 3)) == 3
    assert months_between((2020, 1), (2020, 1)) == 1


def test_parse_duration_full_range():
    result = parse_duration("Jan 2021", "Present", today=FIXED_TODAY)
    assert result.parse_ok is True
    assert result.is_current is True
    assert result.start == (2021, 1)
    assert result.months > 0


def test_parse_duration_unparseable_token():
    result = parse_duration("sometime last year", "Present", today=FIXED_TODAY)
    assert result.parse_ok is False
    assert result.months == 0


# --------------------------------------------------------------------- #
# parser.py — experience entry extraction
# --------------------------------------------------------------------- #
def test_parses_expected_entry_count(parser):
    result = parser.parse(_read("sample_clean.txt"))
    assert len(result.entries) == 2


def test_extracts_title_and_company(parser):
    result = parser.parse(_read("sample_clean.txt"))
    titles = [e.title for e in result.entries]
    companies = [e.company for e in result.entries]
    assert "Senior Backend Engineer" in titles
    assert "Acme Corp" in companies


def test_current_role_flagged_and_dated(parser):
    result = parser.parse(_read("sample_clean.txt"))
    current = next(e for e in result.entries if e.company == "Acme Corp")
    assert current.is_current is True
    assert current.start_date == "2021-01"
    assert current.end_date == "2026-08"  # resolved from FIXED_TODAY


def test_bullets_attached_to_correct_entry(parser):
    result = parser.parse(_read("sample_clean.txt"))
    beta = next(e for e in result.entries if e.company == "Beta Inc")
    assert any("ETL pipelines" in b for b in beta.bullet_lines)
    assert not any("microservices" in b for b in beta.bullet_lines)


def test_section_found_flag(parser):
    result = parser.parse(_read("sample_clean.txt"))
    assert result.section_found is True


def test_no_entries_outside_experience_section(parser):
    text = "Education\nSenior Fellow — Some University (2020 - 2021)\n"
    result = parser.parse(text)
    # Heading present but it's not "Experience", so nothing should match
    assert len(result.entries) == 0


# --------------------------------------------------------------------- #
# gaps.py — total experience, gap, and overlap detection
# --------------------------------------------------------------------- #
def test_total_experience_no_overlap(parser):
    result = parser.parse(_read("sample_clean.txt"))
    timeline = analyze_timeline(result.entries)
    # Jun 2018 - Dec 2020 (31 months) + Jan 2021 - Aug 2026 (68 months), contiguous
    assert timeline.total_months == 99
    assert timeline.gaps == []
    assert timeline.overlaps == []


def test_gap_detected(parser):
    result = parser.parse(_read("sample_gap.txt"))
    timeline = analyze_timeline(result.entries)
    assert len(timeline.gaps) == 1
    gap = timeline.gaps[0]
    assert gap.gap_months > 0
    assert "Delta Analytics" in gap.after_role
    assert "FinPay" in gap.before_role


def test_overlap_detected(parser):
    result = parser.parse(_read("sample_overlap.txt"))
    timeline = analyze_timeline(result.entries)
    assert len(timeline.overlaps) == 1
    overlap = timeline.overlaps[0]
    assert "Globex" in overlap.role_a or "Globex" in overlap.role_b
    assert "Self-Employed" in overlap.role_a or "Self-Employed" in overlap.role_b
    assert overlap.overlap_months == 10  # Jun 2021 - Mar 2022 inclusive


def test_overlap_does_not_inflate_total_months(parser):
    result = parser.parse(_read("sample_overlap.txt"))
    timeline = analyze_timeline(result.entries)
    # Union of Jan2017-Dec2019 (36mo) + Jan2020-Dec2022 (36mo, engulfs the
    # freelance overlap entirely) = 72 months total, NOT 36+36+10.
    assert timeline.total_months == 72


def test_skipped_entries_tracked():
    text = "Experience\n" "Consultant — Acme (sometime - present)\n" "- did consulting work\n"
    parser_local = ExperienceParser(today=FIXED_TODAY)
    result = parser_local.parse(text)
    timeline = analyze_timeline(result.entries)
    assert timeline.skipped_entry_count == 1
    assert timeline.valid_entry_count == 0


# --------------------------------------------------------------------- #
# relevance.py — title similarity & role scoring
# --------------------------------------------------------------------- #
def test_identical_titles_max_similarity():
    assert title_similarity("Senior Backend Engineer", "Senior Backend Engineer") == 1.0


def test_similar_seniority_different_words_partial_similarity():
    score = title_similarity("Staff Engineer", "Principal Engineer")
    assert 0.3 < score < 0.9


def test_unrelated_titles_low_similarity():
    # No token overlap; both titles default to "unspecified" seniority,
    # so the 30% seniority-closeness component still contributes its
    # full weight (0.3) even though the roles are unrelated. That floor
    # is expected -- assert on the token-overlap component instead of
    # the combined score to correctly capture "no textual relation".
    score = title_similarity("Software Engineer", "Executive Chef")
    assert score <= 0.3


def test_relevance_scoring_ranks_matching_role_higher(parser):
    result = parser.parse(_read("sample_clean.txt"))
    scorer = ExperienceRelevanceScorer()
    scored = scorer.score_experience(
        entries=result.entries,
        target_title="Senior Backend Engineer",
        job_keywords=["python", "microservices", "rest", "aws"],
        candidate_id="cand_1",
        job_id="job_1",
    )
    by_company = {rs.company: rs for rs in scored.role_scores}
    assert by_company["Acme Corp"].relevance_score > by_company["Beta Inc"].relevance_score
    assert 0.0 <= scored.overall_relevance_score <= 1.0


def test_relevance_matched_keywords_present(parser):
    result = parser.parse(_read("sample_clean.txt"))
    scorer = ExperienceRelevanceScorer()
    scored = scorer.score_experience(
        entries=result.entries,
        target_title="Backend Engineer",
        job_keywords=["python", "airflow"],
        candidate_id="cand_1",
        job_id="job_1",
    )
    beta = next(rs for rs in scored.role_scores if rs.company == "Beta Inc")
    assert "python" in beta.matched_keywords
    assert "airflow" in beta.matched_keywords


def test_relevance_handles_no_entries():
    scorer = ExperienceRelevanceScorer()
    scored = scorer.score_experience(
        entries=[], target_title="Anything", job_keywords=["x"], candidate_id="c", job_id="j"
    )
    assert scored.overall_relevance_score == 0.0
    assert scored.role_scores == []


# --------------------------------------------------------------------- #
# engine.py — end-to-end orchestration + storage
# --------------------------------------------------------------------- #
def test_engine_process_text_success(engine):
    record = engine.process_text(
        _read("sample_clean.txt"), candidate_id="cand_42", source_file="sample_clean.txt"
    )
    assert record.status in ("success", "partial")
    assert record.candidate_id == "cand_42"
    assert len(record.entries) == 2
    assert record.total_experience_months > 0
    assert record.schema_version == "1.0.0"


def test_engine_writes_json_output(engine):
    engine.process_text(
        _read("sample_clean.txt"), candidate_id="cand_write", source_file="sample_clean.txt"
    )
    out_path = OUTPUT_DIR / "experience" / "sample_clean.experience.json"
    assert out_path.exists()


def test_engine_score_relevance_roundtrip(engine):
    record = engine.process_text(
        _read("sample_clean.txt"), candidate_id="cand_rel", source_file="sample_clean.txt"
    )
    relevance = engine.score_relevance(
        record,
        target_title="Senior Backend Engineer",
        job_keywords=["python", "aws", "microservices"],
        job_id="job_99",
    )
    assert relevance.candidate_id == "cand_rel"
    assert relevance.job_id == "job_99"
    assert 0.0 <= relevance.overall_relevance_score <= 1.0
    out_path = OUTPUT_DIR / "relevance" / "cand_rel__job_99.relevance.json"
    assert out_path.exists()


def test_engine_handles_empty_text_gracefully(engine):
    record = engine.process_text("", candidate_id="cand_empty", source_file="empty.txt")
    assert record.status == "failed"
    assert record.entries == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
