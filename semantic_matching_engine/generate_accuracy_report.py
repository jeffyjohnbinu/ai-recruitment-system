"""
Generate the "Matching accuracy report" deliverable.

Runs the multi-job-type validation set (tests/fixtures/fixtures.py) through
the Semantic Matching Engine, compares predictions against the labeled
ground truth, and writes both a JSON and a Markdown report.

Usage:
    python -m semantic_matching_engine.generate_accuracy_report \
        [--output-dir DIR] \
        [--engine auto|hashing|sentence-transformers]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .matcher import SemanticMatchingEngine
from .reports import ValidationRow, generate_accuracy_report
from .tests.fixtures.fixtures import VALIDATION_PAIRS
from .thresholds import ThresholdConfig, tune_match_threshold


def main():
    parser = argparse.ArgumentParser(description="Generate the semantic matching accuracy report")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--engine", default="auto", choices=["auto", "hashing", "sentence-transformers"]
    )
    args = parser.parse_args()

    # Pass 1: score every validation pair with an engine using default
    # thresholds, purely to collect raw overall_scores against known
    # ground-truth labels (Day 12 task: "Tune similarity thresholds").
    probe_engine = SemanticMatchingEngine(
        output_dir=args.output_dir, embedder_preference=args.engine
    )
    probe_scores, probe_labels = [], []
    for resume, job, _job_type, expected_match in VALIDATION_PAIRS:
        record = probe_engine.match("_probe", "_probe", resume, job, persist=False)
        probe_scores.append(record.overall_score)
        probe_labels.append(expected_match)

    tuning = tune_match_threshold(probe_scores, probe_labels, step=0.01)
    tuned_thresholds = ThresholdConfig(
        match_threshold=tuning.best_threshold,
        strong_match_threshold=min(1.0, tuning.best_threshold + 0.2),
    )
    print(
        f"Tuned match_threshold={tuning.best_threshold:.3f} "
        f"(F1={tuning.best_f1:.3f}, precision={tuning.precision:.3f}, recall={tuning.recall:.3f})\n"
    )

    # Pass 2: re-run with the tuned thresholds and persist the real records.
    engine = SemanticMatchingEngine(
        output_dir=args.output_dir, embedder_preference=args.engine, thresholds=tuned_thresholds
    )

    rows = []
    for i, (resume, job, job_type, expected_match) in enumerate(VALIDATION_PAIRS):
        candidate_id = f"VAL_C{i:02d}_{job_type}"
        job_id = f"VAL_J{i:02d}_{job_type}"
        record = engine.match(candidate_id, job_id, resume, job, persist=True)
        rows.append(
            ValidationRow(
                candidate_id=candidate_id,
                job_id=job_id,
                overall_score=record.overall_score,
                predicted_is_match=record.is_match,
                actual_is_match=expected_match,
                job_type=job_type,
            )
        )

    report = generate_accuracy_report(rows)

    json_path = engine.store.save_report(report.to_dict())
    md_path = (
        Path(args.output_dir) / "structured" / "semantic_matches" / "matching_accuracy_report.md"
    )
    md_path.write_text(report.to_markdown(), encoding="utf-8")

    print(report.to_markdown())
    print(f"\nJSON report written to: {json_path}")
    print(f"Markdown report written to: {md_path}")


if __name__ == "__main__":
    main()
