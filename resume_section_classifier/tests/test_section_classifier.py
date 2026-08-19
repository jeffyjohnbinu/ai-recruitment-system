"""
Automated test suite for the Resume Section Classifier (Day 8).

Run with:
    python -m pytest resume_section_classifier/tests/test_section_classifier.py -v

Or use tests/run_tests.py to also generate a timestamped log file and
the section-detection accuracy report (deliverables).
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from resume_section_classifier.classifier import (  # noqa: E402
    SectionClassifierEngine,
    UnsupportedFileTypeError,
)
from resume_section_classifier.nlp_classifier import classify_block  # noqa: E402
from resume_section_classifier.rules import match_heading  # noqa: E402

FIXTURES_DIR = Path(__file__).parent / "fixtures"
OUTPUT_DIR = Path(__file__).parent / "test_outputs"
REPORTS_DIR = Path(__file__).parent / "reports"

with open(FIXTURES_DIR / "ground_truth.json", encoding="utf-8") as f:
    GROUND_TRUTH = json.load(f)


@pytest.fixture(scope="module")
def engine():
    return SectionClassifierEngine(output_dir=OUTPUT_DIR)


def _record_for(engine, fixture_name):
    text = (FIXTURES_DIR / fixture_name).read_text(encoding="utf-8")
    return engine.classify_text(text, source_name=fixture_name)


def _label_for_substring(record, substring):
    """Find which block a substring lives in, return its label (or None)."""
    for block in record.blocks:
        if substring in block.text:
            return block.label
    return None


# --------------------------------------------------------------------- #
# Rule-based heading detection
# --------------------------------------------------------------------- #
def test_match_heading_recognizes_canonical_forms():
    assert match_heading("Experience") == "Experience"
    assert match_heading("WORK EXPERIENCE") == "Experience"
    assert match_heading("Technical Skills:") == "Skills"
    assert match_heading("Key Projects") == "Projects"


def test_match_heading_rejects_body_text():
    assert match_heading("Built ETL pipelines processing 10TB/day") is None
    assert match_heading("This sentence is definitely not a heading at all") is None


# --------------------------------------------------------------------- #
# Full-heading resume: rule-based path
# --------------------------------------------------------------------- #
def test_full_headings_detects_all_sections(engine):
    record = _record_for(engine, "resume_full_headings.txt")
    for expected in ["Summary", "Experience", "Education", "Skills", "Certifications"]:
        assert expected in record.sections_detected, f"missing section: {expected}"


def test_full_headings_uses_rule_based_method(engine):
    record = _record_for(engine, "resume_full_headings.txt")
    # Every heading-driven block should be rule_heading with full confidence.
    named_blocks = [b for b in record.blocks if b.label != "Header"]
    assert named_blocks
    assert all(b.method == "rule_heading" and b.confidence == 1.0 for b in named_blocks)


def test_full_headings_header_block_detected(engine):
    record = _record_for(engine, "resume_full_headings.txt")
    header_blocks = [b for b in record.blocks if b.label == "Header"]
    assert header_blocks
    assert "john.doe@email.com" in header_blocks[0].text


# --------------------------------------------------------------------- #
# No-heading resume: pure NLP fallback path
# --------------------------------------------------------------------- #
def test_no_headings_falls_back_to_nlp(engine):
    record = _record_for(engine, "resume_no_headings.txt")
    assert record.heading_based_blocks == 0
    assert record.nlp_based_blocks > 0


def test_no_headings_still_finds_experience_and_education(engine):
    record = _record_for(engine, "resume_no_headings.txt")
    assert "Experience" in record.sections_detected
    assert "Education" in record.sections_detected


# --------------------------------------------------------------------- #
# Partial-heading resume: mixed rule + NLP
# --------------------------------------------------------------------- #
def test_partial_headings_uses_both_methods(engine):
    record = _record_for(engine, "resume_partial_headings.txt")
    methods = {b.method for b in record.blocks}
    assert "rule_heading" in methods
    assert "nlp" in methods


def test_partial_headings_classifies_unlabeled_experience(engine):
    record = _record_for(engine, "resume_partial_headings.txt")
    label = _label_for_substring(record, "Owned roadmap for payments platform")
    assert label == "Experience"


# --------------------------------------------------------------------- #
# Table-linearized skills content (mirrors Day 5's pipe-separated rows)
# --------------------------------------------------------------------- #
def test_table_linearized_skills_stay_under_skills_heading(engine):
    record = _record_for(engine, "resume_table_skills.txt")
    label = _label_for_substring(record, "Other | Scikit-learn, Pandas, XGBoost")
    assert label == "Skills"


# --------------------------------------------------------------------- #
# Heading / content disagreement flagging
# --------------------------------------------------------------------- #
def test_mismatched_heading_is_flagged():
    engine2 = SectionClassifierEngine(output_dir=OUTPUT_DIR)
    # "Skills" heading over content that reads entirely as work experience.
    text = (
        "Skills\n"
        "Managed a team of 5 engineers and led migration to microservices, "
        "shipping 12 features across 3 quarters while mentoring junior staff.\n"
    )
    record = engine2.classify_text(text, source_name="mismatch_test")
    assert any("heading says" in w for w in record.warnings)


# --------------------------------------------------------------------- #
# NLP classifier unit behavior
# --------------------------------------------------------------------- #
def test_nlp_classifier_scores_skills_block():
    result = classify_block("Proficient in Python, SQL, AWS, Docker, and Kubernetes.")
    assert result.label == "Skills"


def test_nlp_classifier_scores_education_block():
    result = classify_block("B.S. in Computer Science, University of Texas, GPA 3.9")
    assert result.label == "Education"


def test_nlp_classifier_low_signal_is_uncategorized():
    result = classify_block("Lorem ipsum dolor sit amet consectetur adipiscing elit.")
    assert result.label == "Uncategorized"


# --------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------- #
def test_unsupported_extension_raises(engine, tmp_path):
    bad_file = tmp_path / "resume.xyz"
    bad_file.write_text("not a real resume")
    with pytest.raises(UnsupportedFileTypeError):
        engine.process_file(bad_file)


def test_missing_file_raises(engine):
    with pytest.raises(FileNotFoundError):
        engine.process_file(FIXTURES_DIR / "does_not_exist.txt")


def test_empty_text_marked_failed(engine):
    record = engine.classify_text("   \n  ", source_name="empty")
    assert record.status == "failed"


# --------------------------------------------------------------------- #
# Section detection accuracy report (also a deliverable)
# --------------------------------------------------------------------- #
def test_generate_accuracy_report(engine):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Section Detection Accuracy Report",
        "",
        "Day 8 deliverable — Zecpath AI Job Portal",
        "",
        "Ground truth: hand-labeled substrings across 5 fixture resumes "
        "(full headings, no headings, partial headings, table-linearized "
        "skills, noisy/OCR-style spacing).",
        "",
        "| Fixture | Correct | Total | Accuracy |",
        "|---|---|---|---|",
    ]

    total_correct = 0
    total_count = 0
    per_fixture_rows = []

    for fixture_name, cases in GROUND_TRUTH.items():
        record = _record_for(engine, fixture_name)
        correct = 0
        mismatches = []
        for case in cases:
            predicted = _label_for_substring(record, case["contains"])
            if predicted == case["expected_label"]:
                correct += 1
            else:
                mismatches.append((case["contains"], case["expected_label"], predicted))
        total = len(cases)
        total_correct += correct
        total_count += total
        acc = correct / total if total else 0.0
        per_fixture_rows.append((fixture_name, correct, total, acc, mismatches))
        lines.append(f"| {fixture_name} | {correct} | {total} | {acc:.0%} |")

    overall = total_correct / total_count if total_count else 0.0
    lines.append(f"| **Overall** | **{total_correct}** | **{total_count}** | **{overall:.0%}** |")
    lines.append("")

    any_mismatch = False
    for fixture_name, correct, total, acc, mismatches in per_fixture_rows:
        if mismatches:
            any_mismatch = True
            lines.append(f"### Mismatches in `{fixture_name}`")
            for snippet, expected, predicted in mismatches:
                lines.append(f"- `{snippet[:50]}...` expected **{expected}**, got **{predicted}**")
            lines.append("")

    if not any_mismatch:
        lines.append("No mismatches — all labeled ground-truth snippets classified correctly.")
        lines.append("")

    report_path = REPORTS_DIR / "accuracy_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # Regression guard: keep the bar high as the lexicon/rules evolve.
    assert overall >= 0.85, f"Section detection accuracy dropped to {overall:.0%}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
