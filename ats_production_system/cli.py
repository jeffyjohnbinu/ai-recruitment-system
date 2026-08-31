"""
Command-line entry point for the Day 20 ATS Production System.

Usage:
    python -m ats_production_system.cli \
        --resumes demo_datasets/resumes/ \
        --job demo_datasets/job_descriptions/senior_backend_engineer.txt \
        --job-id job_senior_backend \
        --role "senior_backend_engineer" \
        --output-dir outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import ATSPipeline

SUPPORTED_RESUME_EXTENSIONS = {".pdf", ".docx"}


def main() -> None:
    parser = argparse.ArgumentParser(description="ATS Production System (Day 20)")
    parser.add_argument("--resumes", required=True, help="Directory of resume files (.pdf/.docx)")
    parser.add_argument("--job", required=True, help="Path to a job description file")
    parser.add_argument("--job-id", required=True, help="Identifier for this job posting")
    parser.add_argument("--role", default=None, help="Weight-profile role key (optional)")
    parser.add_argument("--output-dir", default="outputs", help="Where to write structured JSON")
    args = parser.parse_args()

    resumes_dir = Path(args.resumes)
    if not resumes_dir.is_dir():
        print(f"Not a directory: {resumes_dir}", file=sys.stderr)
        sys.exit(1)

    resume_paths = sorted(
        p for p in resumes_dir.iterdir() if p.suffix.lower() in SUPPORTED_RESUME_EXTENSIONS
    )
    if not resume_paths:
        print(f"No .pdf/.docx resumes found in {resumes_dir}", file=sys.stderr)
        sys.exit(1)

    pipeline = ATSPipeline(output_dir=args.output_dir, embedder_preference="auto")

    print("Engine availability:")
    for name, available in pipeline.engine_status().items():
        print(f"  [{'OK' if available else 'MISSING'}] {name}")
    print()

    result = pipeline.run_batch(
        resume_paths=resume_paths, job_path=args.job, job_id=args.job_id, role=args.role
    )

    summary = result["summary"]
    print(f"Job: {args.job_id}")
    print(f"Total candidates : {summary.total_candidates}")
    print(f"Shortlist        : {summary.shortlist_count}")
    print(f"Review           : {summary.review_count}")
    print(f"Auto-reject      : {summary.auto_reject_count}")
    print()
    print(f"{'Candidate':<30} {'Score':>7}  {'Zone':<12} Status")
    print("-" * 70)
    for c in sorted(summary.candidates, key=lambda x: (x["final_score"] or 0), reverse=True):
        score_str = f"{c['final_score']:.2f}" if c["final_score"] is not None else "  n/a"
        print(f"{c['candidate_id']:<30} {score_str:>7}  {(c['zone'] or '-'): <12} {c['status']}")

    print(f"\nStructured output written to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
