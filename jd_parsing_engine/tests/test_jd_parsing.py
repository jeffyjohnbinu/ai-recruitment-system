"""
Automated test suite for the Job Description Parsing System.

Run with:
    python -m pytest jd_parsing_engine/tests/test_jd_parsing.py -v

Or use tests/run_tests.py to also generate a timestamped log file
(deliverable: "Test result logs").
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jd_parsing_engine.cleaner import JDTextCleaner  # noqa: E402
from jd_parsing_engine.extractors import (  # noqa: E402
    extract_education,
    extract_experience,
    extract_role,
    extract_skills,
)
from jd_parsing_engine.parser import JDParsingEngine, UnsupportedFileTypeError  # noqa: E402

SAMPLES_DIR = Path(__file__).parent / "sample_jds"
OUTPUT_DIR = Path(__file__).parent / "test_outputs"


@pytest.fixture(scope="module")
def engine():
    return JDParsingEngine(output_dir=OUTPUT_DIR)


# --------------------------------------------------------------------- #
# End-to-end: sample_jd_1_backend.txt (clean, well-structured)
# --------------------------------------------------------------------- #
def test_backend_jd_parses_successfully(engine):
    record = engine.process_file(SAMPLES_DIR / "sample_jd_1_backend.txt")
    assert record.status in ("success", "partial")
    assert record.cleaned_char_count > 0


def test_backend_jd_detects_role(engine):
    record = engine.process_file(SAMPLES_DIR / "sample_jd_1_backend.txt")
    assert record.normalized_role == "Backend Engineer"
    assert record.seniority_level == "Senior"


def test_backend_jd_detects_required_skills(engine):
    record = engine.process_file(SAMPLES_DIR / "sample_jd_1_backend.txt")
    for skill in ["Python", "SQL", "Docker", "Kubernetes", "CI/CD"]:
        assert skill in record.required_skills, f"missing required skill: {skill}"


def test_backend_jd_detects_preferred_skills(engine):
    record = engine.process_file(SAMPLES_DIR / "sample_jd_1_backend.txt")
    assert "Kafka" in record.preferred_skills
    assert "Terraform" in record.preferred_skills
    # Preferred skills should not duplicate required ones
    assert not set(record.preferred_skills) & set(record.required_skills)


def test_backend_jd_detects_experience_years(engine):
    record = engine.process_file(SAMPLES_DIR / "sample_jd_1_backend.txt")
    assert record.min_experience_years == 5.0


def test_backend_jd_detects_education(engine):
    record = engine.process_file(SAMPLES_DIR / "sample_jd_1_backend.txt")
    assert record.education_level == "Bachelor's"
    assert "Computer Science" in record.education_fields


def test_backend_jd_drops_eeo_boilerplate(engine):
    record = engine.process_file(SAMPLES_DIR / "sample_jd_1_backend.txt")
    assert "equal opportunity" not in record.cleaned_text.lower()


def test_backend_jd_normalizes_bullets(engine):
    record = engine.process_file(SAMPLES_DIR / "sample_jd_1_backend.txt")
    lines = [ln for ln in record.cleaned_text.splitlines() if "REST APIs" in ln]
    assert lines
    assert lines[0].startswith("- ")


# --------------------------------------------------------------------- #
# End-to-end: sample_jd_2_pm.txt (experience range, dual degree mention)
# --------------------------------------------------------------------- #
def test_pm_jd_detects_role_from_label(engine):
    record = engine.process_file(SAMPLES_DIR / "sample_jd_2_pm.txt")
    assert record.raw_title == "Associate Product Manager"
    assert record.normalized_role == "Product Manager"


def test_pm_jd_detects_experience_range(engine):
    record = engine.process_file(SAMPLES_DIR / "sample_jd_2_pm.txt")
    assert record.min_experience_years == 2.0
    assert record.max_experience_years == 4.0


def test_pm_jd_detects_skills(engine):
    record = engine.process_file(SAMPLES_DIR / "sample_jd_2_pm.txt")
    assert "SQL" in record.required_skills
    assert "Figma" in record.preferred_skills


def test_pm_jd_drops_apply_now_footer(engine):
    record = engine.process_file(SAMPLES_DIR / "sample_jd_2_pm.txt")
    assert "apply now" not in record.cleaned_text.lower()


# --------------------------------------------------------------------- #
# End-to-end: sample_jd_3_noisy_datasci.txt (noisy formatting, PhD/Master's)
# --------------------------------------------------------------------- #
def test_noisy_jd_detects_role(engine):
    record = engine.process_file(SAMPLES_DIR / "sample_jd_3_noisy_datasci.txt")
    assert record.normalized_role == "Data Scientist"


def test_noisy_jd_detects_experience(engine):
    record = engine.process_file(SAMPLES_DIR / "sample_jd_3_noisy_datasci.txt")
    assert record.min_experience_years == 3.0


def test_noisy_jd_detects_ml_skills(engine):
    record = engine.process_file(SAMPLES_DIR / "sample_jd_3_noisy_datasci.txt")
    assert "Machine Learning" in record.required_skills
    assert "Python" in record.required_skills
    assert "Pandas" in record.required_skills


def test_noisy_jd_drops_job_id_and_posted_lines(engine):
    record = engine.process_file(SAMPLES_DIR / "sample_jd_3_noisy_datasci.txt")
    lower = record.cleaned_text.lower()
    assert "job id" not in lower
    assert "posted on" not in lower


def test_noisy_jd_drops_bare_page_number(engine):
    record = engine.process_file(SAMPLES_DIR / "sample_jd_3_noisy_datasci.txt")
    lines = [ln.strip() for ln in record.cleaned_text.splitlines()]
    assert "1" not in lines


def test_noisy_jd_education_prefers_first_match(engine):
    record = engine.process_file(SAMPLES_DIR / "sample_jd_3_noisy_datasci.txt")
    assert record.education_level == "PhD"


# --------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------- #
def test_unsupported_extension_raises(engine, tmp_path):
    bad_file = tmp_path / "job.pdf"
    bad_file.write_bytes(b"%PDF-1.4 not really a pdf")
    with pytest.raises(UnsupportedFileTypeError):
        engine.process_file(bad_file)


def test_missing_file_raises(engine):
    with pytest.raises(FileNotFoundError):
        engine.process_file(SAMPLES_DIR / "does_not_exist.txt")


def test_process_text_in_memory(engine):
    record = engine.process_text(
        "Data Analyst\nRequirements\n- 2+ years experience\n- SQL and Excel required\n"
        "- Bachelor's degree in Statistics",
        source_name="inline_jd",
    )
    assert record.normalized_role == "Data Analyst"
    assert "SQL" in record.required_skills
    assert "Excel" in record.required_skills
    assert record.min_experience_years == 2.0


# --------------------------------------------------------------------- #
# AI profile output
# --------------------------------------------------------------------- #
def test_ai_profile_contains_expected_keys(engine):
    record = engine.process_file(SAMPLES_DIR / "sample_jd_1_backend.txt")
    profile = record.to_ai_profile()
    for key in [
        "role",
        "seniority_level",
        "required_skills",
        "preferred_skills",
        "min_experience_years",
        "max_experience_years",
        "education_level",
        "education_fields",
    ]:
        assert key in profile


# --------------------------------------------------------------------- #
# Cleaner / extractor unit tests (isolated from file I/O)
# --------------------------------------------------------------------- #
def test_cleaner_normalizes_heading_variants():
    cleaner = JDTextCleaner()
    text, _ = cleaner.clean("What You'll Need\nSome text\nnice to have\nOther text")
    assert "Requirements" in text
    assert "Preferred Qualifications" in text


def test_cleaner_unifies_bullet_styles():
    cleaner = JDTextCleaner()
    text, report = cleaner.clean("● First point\n▪ Second point\n* Third point")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert all(ln.startswith("- ") for ln in lines)
    assert report.normalized_bullets == 3


def test_extract_experience_handles_range():
    info = extract_experience("Looking for someone with 3-5 years of experience.")
    assert info.min_years == 3.0
    assert info.max_years == 5.0


def test_extract_experience_handles_plus():
    info = extract_experience("7+ years in software development required.")
    assert info.min_years == 7.0
    assert info.max_years is None


def test_extract_role_normalizes_swe_abbreviation():
    info = extract_role("SWE II\nRequirements\n- Python")
    assert info.normalized_role == "Software Engineer"


def test_extract_skills_deduplicates_across_scopes():
    required_scope = "Python, SQL, and AWS required."
    preferred_scope = "Python is a plus but not required."
    info = extract_skills(
        "Python, SQL, and AWS required.\nPython is a plus but not required.",
        required_scope,
        preferred_scope,
    )
    assert "Python" in info.required_skills
    assert "Python" not in info.preferred_skills


def test_extract_education_detects_field_of_study():
    info = extract_education(
        "Bachelor's degree in Computer Science required.",
        "Bachelor's degree in Computer Science required.",
    )
    assert info.degree_level == "Bachelor's"
    assert "Computer Science" in info.fields_of_study


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
