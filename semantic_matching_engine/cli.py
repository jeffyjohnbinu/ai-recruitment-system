"""
Command-line entry point.

Usage:
    python -m semantic_matching_engine.cli \
        --resume RESUME.json \
        --job JOB.json \
        --candidate-id C001 \
        --job-id J001 \
        [--output-dir OUTPUT_DIR] \
        [--engine auto|hashing|sentence-transformers]

Both --resume and --job accept a path to a JSON file containing either:
  - {"skills": "...", "experience": "...", "projects": "..."}, or
  - a Day 8 section-classifier map, or
  - a Day 6 JobRequirementRecord (for --job)

Examples:
    python -m semantic_matching_engine.cli --resume resume_sections.json \
        --job job_requirements.json --candidate-id C001 --job-id J001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .matcher import SemanticMatchingEngine


def main():
    parser = argparse.ArgumentParser(description="Semantic Resume <-> Job Matching Engine")
    parser.add_argument("--resume", required=True, help="Path to a JSON file with resume data")
    parser.add_argument(
        "--job", required=True, help="Path to a JSON file with job description data"
    )
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output-dir", default="outputs", help="Where to write structured JSON")
    parser.add_argument(
        "--engine",
        default="auto",
        choices=["auto", "hashing", "sentence-transformers"],
        help="Embedding engine to use",
    )
    args = parser.parse_args()

    resume_path = Path(args.resume)
    job_path = Path(args.job)

    for path in (resume_path, job_path):
        if not path.exists():
            print(f"Path not found: {path}", file=sys.stderr)
            sys.exit(1)

    resume_input = json.loads(resume_path.read_text(encoding="utf-8"))
    job_input = json.loads(job_path.read_text(encoding="utf-8"))

    engine = SemanticMatchingEngine(
        output_dir=args.output_dir,
        embedder_preference=args.engine,
    )
    record = engine.match(
        candidate_id=args.candidate_id,
        job_id=args.job_id,
        resume_input=resume_input,
        job_input=job_input,
    )

    print(f"\nMatch result: {record.candidate_id} <-> {record.job_id}")
    print(f"  Engine used     : {record.engine_used}")
    print(f"  Overall score   : {record.overall_score:.4f}")
    print(f"  Match band      : {record.match_band}")
    print(f"  Is match        : {record.is_match}")
    print(f"  Status          : {record.status}")
    if record.warnings:
        print(f"  Warnings        : {'; '.join(record.warnings)}")
    print(f"\nOutput written to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
