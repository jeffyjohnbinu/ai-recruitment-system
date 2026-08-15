"""
Automated test suite for the Resume Text Extraction Engine.

Run with:
    python -m pytest resume_extraction_engine/tests/test_extraction.py -v

Or use tests/run_tests.py to also generate a timestamped log file
(deliverable: "Test result logs").
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from resume_extraction_engine.cleaner import TextCleaner, TextNormalizer  # noqa: E402
from resume_extraction_engine.extractor import (  # noqa: E402
    ResumeExtractionEngine,
    UnsupportedFileTypeError,
)

SAMPLES_DIR = Path(__file__).parent / "sample_resumes"
OUTPUT_DIR = Path(__file__).parent / "test_outputs"


@pytest.fixture(scope="module")
def engine():
    return ResumeExtractionEngine(output_dir=OUTPUT_DIR)


# --------------------------------------------------------------------- #
# DOCX extraction
# --------------------------------------------------------------------- #
def test_docx_extracts_text(engine):
    record = engine.process_file(SAMPLES_DIR / "sample_resume_1.docx")
    assert record.status in ("success", "partial")
    assert record.cleaned_char_count > 0
    assert "John" in record.cleaned_text or "JOHN" in record.cleaned_text


def test_docx_detects_sections(engine):
    record = engine.process_file(SAMPLES_DIR / "sample_resume_1.docx")
    for expected in ["Summary", "Experience", "Education", "Skills", "Certifications"]:
        assert expected in record.sections_detected, f"missing section: {expected}"


def test_docx_extracts_table_content(engine):
    record = engine.process_file(SAMPLES_DIR / "sample_resume_1.docx")
    assert record.tables_found >= 1
    assert "Kubernetes" in record.cleaned_text


def test_docx_normalizes_bullets(engine):
    record = engine.process_file(SAMPLES_DIR / "sample_resume_1.docx")
    # Original bullets were ●, •, ▪ -- all should become "- "
    lines_with_content = [ln for ln in record.cleaned_text.splitlines() if "Led migration" in ln]
    assert lines_with_content, "expected bullet line not found"
    assert lines_with_content[0].startswith("- ")


def test_docx_shouting_summary_gets_title_cased(engine):
    record = engine.process_file(SAMPLES_DIR / "sample_resume_1.docx")
    assert "RESULTS-DRIVEN SOFTWARE ENGINEER WITH 6 YEARS" not in record.cleaned_text
    assert "Results-driven" in record.cleaned_text or "Results-Driven" in record.cleaned_text


# --------------------------------------------------------------------- #
# PDF extraction — two-column layout
# --------------------------------------------------------------------- #
def test_pdf_twocolumn_extracts_both_columns(engine):
    record = engine.process_file(SAMPLES_DIR / "sample_resume_2_twocolumn.pdf")
    assert record.status in ("success", "partial")
    assert "Product Manager" in record.cleaned_text
    assert "Roadmapping" in record.cleaned_text


def test_pdf_twocolumn_does_not_interleave_lines(engine):
    """
    Regression guard: raw left-to-right pdfplumber extraction on a true
    two-column layout would scramble 'Product Manager' with skill words
    on the same line. Confirm columns were separated.
    """
    record = engine.process_file(SAMPLES_DIR / "sample_resume_2_twocolumn.pdf")
    for line in record.cleaned_text.splitlines():
        # A scrambled line would contain both an experience phrase and a
        # skill phrase on the same line -- that should not happen here.
        if "Product Manager" in line:
            assert "Roadmapping" not in line


# --------------------------------------------------------------------- #
# PDF extraction — noisy single column
# --------------------------------------------------------------------- #
def test_pdf_noisy_dehyphenates(engine):
    record = engine.process_file(SAMPLES_DIR / "sample_resume_3_noisy.pdf")
    assert (
        "Machine Learning" in record.cleaned_text
        or "MACHINE LEARNING" in record.cleaned_text.upper()
    )
    assert "Learn-\nIng" not in record.cleaned_text
    assert "Learn-\ning" not in record.cleaned_text


def test_pdf_noisy_drops_page_number_lines(engine):
    record = engine.process_file(SAMPLES_DIR / "sample_resume_3_noisy.pdf")
    lines = [ln.strip() for ln in record.cleaned_text.splitlines()]
    assert "1" not in lines, "bare page-number line should have been dropped"


def test_pdf_noisy_normalizes_bullet(engine):
    record = engine.process_file(SAMPLES_DIR / "sample_resume_3_noisy.pdf")
    fraud_lines = [ln for ln in record.cleaned_text.splitlines() if "fraud detection" in ln]
    assert fraud_lines
    assert fraud_lines[0].startswith("- ")


# --------------------------------------------------------------------- #
# Error handling / unsupported types
# --------------------------------------------------------------------- #
def test_unsupported_extension_raises(engine, tmp_path):
    bad_file = tmp_path / "resume.txt"
    bad_file.write_text("not a real resume")
    with pytest.raises(UnsupportedFileTypeError):
        engine.process_file(bad_file)


def test_missing_file_raises(engine):
    with pytest.raises(FileNotFoundError):
        engine.process_file(SAMPLES_DIR / "does_not_exist.pdf")


# --------------------------------------------------------------------- #
# Cleaner / normalizer unit tests (isolated from file I/O)
# --------------------------------------------------------------------- #
def test_cleaner_collapses_blank_lines():
    cleaner = TextCleaner()
    text, report = cleaner.clean("Line one\n\n\n\n\nLine two")
    assert "\n\n\n" not in text
    assert report.collapsed_blank_lines >= 1


def test_cleaner_unifies_bullet_styles():
    cleaner = TextCleaner()
    text, report = cleaner.clean("● First point\n▪ Second point\n* Third point")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert all(ln.startswith("- ") for ln in lines)
    assert report.normalized_bullets == 3


def test_cleaner_normalizes_heading_variants():
    cleaner = TextCleaner()
    text, _ = cleaner.clean("WORK EXPERIENCE\nSome job\ntechnical skills\nPython")
    assert "Experience" in text
    assert "Skills" in text


def test_normalizer_preserves_short_acronyms():
    normalizer = TextNormalizer()
    out = normalizer.normalize_capitalization(
        "CERTIFIED AWS SOLUTIONS ARCHITECT WITH SEVEN YEARS OF HANDS ON EXPERIENCE"
    )
    # long shouted sentence -> title-cased, but short acronym-like tokens kept as-is
    assert out != "CERTIFIED AWS SOLUTIONS ARCHITECT WITH SEVEN YEARS OF HANDS ON EXPERIENCE"
    assert "AWS" in out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
