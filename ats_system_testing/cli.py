"""
Command-line entry point.

Usage:
    python -m ats_system_testing.cli [--output-dir OUTPUT_DIR]

Runs the full Day 17 fixture set through the ATS pipeline, prints a
summary table, computes accuracy metrics, derives an improvement
backlog, and writes a structured JSON result file.
"""

from __future__ import annotations

import argparse
import sys

from .backlog import build_backlog
from .harness import ATSTestHarness
from .metrics import compute_metrics
from .storage import ResultStore


def main():
    parser = argparse.ArgumentParser(description="ATS System Testing (Day 17)")
    parser.add_argument(
        "--output-dir", default="outputs", help="Where to write the structured JSON result"
    )
    args = parser.parse_args()

    harness = ATSTestHarness()
    results = harness.run()
    metrics = compute_metrics(results)
    backlog = build_backlog(metrics)

    print("\n--- ATS System Testing (Day 17) ---")
    header = f"{'Case ID':<24}{'Role':<10}{'Seniority':<10}{'AI rec.':<10}{'Human rec.':<12}"
    print(header + "Shortlist match")
    for r in results:
        match = "OK" if r.is_correct else "MISMATCH"
        print(
            f"{r.test_case.case_id:<24}{r.test_case.role_type:<10}"
            f"{r.test_case.seniority:<10}{r.pipeline_output.recommendation:<10}"
            f"{r.test_case.ground_truth.recommendation:<12}{match}"
        )

    overall = metrics.overall
    print("\n--- Accuracy Metrics ---")
    print(
        f"Accuracy: {overall.accuracy:.2%}  Precision: {overall.precision:.2%}  "
        f"Recall: {overall.recall:.2%}  F1: {overall.f1:.2%}  (n={overall.total})"
    )

    if metrics.mismatches:
        print(f"\n{len(metrics.mismatches)} mismatch(es) found:")
        for m in metrics.mismatches:
            print(f"  [{m.mismatch_type}] {m.case_id}: {m.notes}")
    else:
        print("\nNo mismatches -- AI output matched manual review on every case.")

    print(f"\n--- Improvement Backlog ({len(backlog)} item(s)) ---")
    for item in backlog:
        print(f"  [{item.priority}] {item.title}")

    store = ResultStore(args.output_dir)
    path = store.save(results, metrics)
    print(f"\nStructured result written to: {path.resolve()}")


if __name__ == "__main__":
    sys.exit(main() or 0)
