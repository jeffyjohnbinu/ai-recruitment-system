"""
Automated test suite for the Education & Certification Parsing Engine.

Run with:
    python -m pytest education_certification_extractor/tests/test_extraction.py -v

Or use tests/run_tests.py to also generate a timestamped log file.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from education_certification_extractor.certification_parser import CertificationParser  # noqa: E402
from education_certification_extractor.degree_parser import DegreeParser  # noqa: E402
from education_certification_extractor.extractor import (  # noqa: E402
    EducationCertificationExtractor,
)
from education_certification_extractor.normalizer import (  # noqa: E402
    normalize_certification_name,
    normalize_degree,
    normalize_institution,
)
from education_certification_extractor.relevance_tagger import tag_relevance  # noqa: E402

SAMPLES_DIR = Path(__file__).parent / "sample_data"
OUTPUT_DIR = Path(__file__).parent / "test_outputs"


@pytest.fixture(scope="module")
def engine():
    return EducationCertificationExtractor(output_dir=OUTPUT_DIR)


def read_fixture(name: str) -> str:
    return (SAMPLES_DIR / name).read_text(encoding="utf-8")


# --------------------------------------------------------------------- #
# Normalizer unit tests
# --------------------------------------------------------------------- #
def test_normalize_degree_recognizes_common_abbreviations():
    assert normalize_degree("B.S.") == "Bachelor of Science"
    assert normalize_degree("BTech") == "Bachelor of Technology"
    assert normalize_degree("MBA") == "Master of Business Administration"
    assert normalize_degree("PhD") == "Doctor of Philosophy"
    assert normalize_degree("Ph.D.") == "Doctor of Philosophy"


def test_normalize_degree_unrecognized_returns_none():
    assert normalize_degree("Certificate of Attendance") is None
    assert normalize_degree("") is None


def test_normalize_institution_strips_articles_and_punctuation():
    assert (
        normalize_institution("  the University of Texas at Austin, ")
        == "University of Texas at Austin"
    )


def test_normalize_certification_name_collapses_whitespace():
    assert (
        normalize_certification_name("AWS  Certified   Solutions Architect -")
        == "AWS Certified Solutions Architect"
    )


# --------------------------------------------------------------------- #
# Relevance tagging unit tests
# --------------------------------------------------------------------- #
def test_tag_relevance_cloud():
    assert tag_relevance("AWS Certified Solutions Architect") == "Cloud"


def test_tag_relevance_project_management():
    assert tag_relevance("Project Management Professional (PMP)") == "Project Management"


def test_tag_relevance_security():
    assert tag_relevance("CISSP") == "Security"


def test_tag_relevance_unknown_falls_back_to_other():
    assert tag_relevance("Random Internal Training Badge") == "Other"


# --------------------------------------------------------------------- #
# Degree parser unit tests
# --------------------------------------------------------------------- #
def test_degree_parser_full_line():
    parser = DegreeParser()
    records = parser.parse_section("B.S. in Computer Science — University of Texas at Austin, 2018")
    assert len(records) == 1
    rec = records[0]
    assert rec.degree_type_normalized == "Bachelor of Science"
    assert rec.field_of_study == "Computer Science"
    assert rec.institution == "University of Texas at Austin"
    assert rec.graduation_year == 2018
    assert rec.confidence >= 0.9


def test_degree_parser_multiple_lines():
    parser = DegreeParser()
    text = (
        "M.Tech in Data Science -- IIT Bombay, 2016\n"
        "B.Tech in Computer Engineering -- VJTI Mumbai, 2013"
    )
    records = parser.parse_section(text)
    assert len(records) == 2
    assert records[0].degree_type_normalized == "Master of Technology"
    assert records[1].degree_type_normalized == "Bachelor of Technology"


def test_degree_parser_handles_split_across_two_lines():
    """PhD listed on one line, institution/year on the next -- common template."""
    parser = DegreeParser()
    text = "PhD in Machine Learning\nCarnegie Mellon University, 2019"
    records = parser.parse_section(text)
    assert len(records) >= 1
    phd_records = [r for r in records if r.degree_type_normalized == "Doctor of Philosophy"]
    assert phd_records
    assert phd_records[0].field_of_study and "Machine Learning" in phd_records[0].field_of_study


def test_degree_parser_skips_heading_line():
    parser = DegreeParser()
    records = parser.parse_section("Education\nB.S. in Computer Science — UT Austin, 2018")
    assert len(records) == 1


def test_degree_parser_empty_section_returns_empty_list():
    parser = DegreeParser()
    assert parser.parse_section("") == []
    assert parser.parse_section("   \n  ") == []


# --------------------------------------------------------------------- #
# Certification parser unit tests
# --------------------------------------------------------------------- #
def test_certification_parser_basic_line():
    parser = CertificationParser()
    records = parser.parse_section("- AWS Certified Solutions Architect – Associate")
    assert len(records) == 1
    assert records[0].relevance_category == "Cloud"


def test_certification_parser_extracts_year_and_issuer():
    parser = CertificationParser()
    records = parser.parse_section("- Project Management Professional (PMP) -- PMI, 2021")
    assert len(records) == 1
    rec = records[0]
    assert rec.year == 2021
    assert rec.relevance_category == "Project Management"


def test_certification_parser_multiple_lines():
    parser = CertificationParser()
    text = (
        "- AWS Certified Solutions Architect -- Professional, 2023\n"
        "- Certified Scrum Master -- Scrum Alliance\n"
        "- Six Sigma Green Belt"
    )
    records = parser.parse_section(text)
    assert len(records) == 3
    categories = {r.relevance_category for r in records}
    assert "Cloud" in categories
    assert "Agile & Scrum" in categories
    assert "Quality & Process" in categories


def test_certification_parser_empty_section_returns_empty_list():
    parser = CertificationParser()
    assert parser.parse_section("") == []


# --------------------------------------------------------------------- #
# End-to-end extractor tests (via extract_from_text)
# --------------------------------------------------------------------- #
def test_extract_from_text_success(engine):
    text = read_fixture("resume_1_clean.txt")
    record = engine.extract_from_text(text, source_file="resume_1_clean.txt")
    assert record.status in ("success", "partial")
    assert record.degrees_found == 1
    assert record.certifications_found == 1
    assert record.highest_degree == "Bachelor of Science"


def test_extract_from_text_multi_degree(engine):
    text = read_fixture("resume_2_multi_degree.txt")
    record = engine.extract_from_text(text, source_file="resume_2_multi_degree.txt")
    assert record.degrees_found == 2
    assert record.certifications_found == 4
    # Master's degree should outrank Bachelor's for highest_degree
    assert record.highest_degree == "Master of Technology"


def test_extract_from_text_noisy_education(engine):
    text = read_fixture("resume_3_noisy_education.txt")
    record = engine.extract_from_text(text, source_file="resume_3_noisy_education.txt")
    assert record.degrees_found >= 2
    assert record.highest_degree == "Doctor of Philosophy"
    cert_names = [c.name for c in record.certifications]
    assert any("CISSP" in n for n in cert_names)


def test_extract_from_sections_matches_alias_headings(engine):
    record = engine.extract_from_sections(
        {
            "Academic Background": "B.S. in Computer Science — UT Austin, 2018",
            "Licenses & Certifications": "- AWS Certified Solutions Architect",
        },
        source_file="alias_test",
    )
    assert record.degrees_found == 1
    assert record.certifications_found == 1


def test_extract_from_text_missing_sections_flags_warnings(engine):
    record = engine.extract_from_text(
        "JANE DOE\nSummary\nGeneralist with broad experience.\n",
        source_file="no_education.txt",
    )
    assert record.status == "failed"
    assert record.degrees_found == 0
    assert record.certifications_found == 0
    assert record.warnings


def test_extract_writes_json_output(engine):
    text = read_fixture("resume_1_clean.txt")
    engine.extract_from_text(text, source_file="resume_1_clean.txt")
    output_file = OUTPUT_DIR / "structured" / "resume_1_clean.academic_profile.json"
    assert output_file.exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
