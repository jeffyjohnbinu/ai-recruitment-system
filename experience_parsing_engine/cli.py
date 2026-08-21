"""
Command-line entry point.

Usage:
    python -m experience_parsing_engine.cli <resume_text_file> --candidate-id ID [options]

Examples:
    python -m experience_parsing_engine.cli resume.clean.txt --candidate-id cand_001

    python -m experience_parsing_engine.cli resume.clean.txt --candidate-id cand_001 \\
        --target-title "Senior Backend Engineer" \\
        --job-keywords python aws microservices \\
        --job-id job_001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .engine import ExperienceParsingEngine


def main():
    parser = argparse.ArgumentParser(description="Experience Parsing & Relevance Engine")
    parser.add_argument("resume_text_file", help="Path to a cleaned resume .txt file")
    parser.add_argument("--candidate-id", required=True, help="Candidate identifier")
    parser.add_argument("--output-dir", default="outputs", help="Where to write structured JSON")
    parser.add_argument("--target-title", help="Job title to score relevance against")
    parser.add_argument("--job-keywords", nargs="*", default=[], help="Job keyword list")
    parser.add_argument(
        "--job-id", default="job_unknown", help="Job identifier for relevance output"
    )
    args = parser.parse_args()

    path = Path(args.resume_text_file)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    text = path.read_text(encoding="utf-8")
    engine = ExperienceParsingEngine(output_dir=args.output_dir)
    record = engine.process_text(text, candidate_id=args.candidate_id, source_file=str(path))

    print(f"\nStatus: {record.status}")
    print(f"Entries found: {len(record.entries)}")
    print(
        f"Total experience: {record.total_experience_years} years "
        f"({record.total_experience_months} months)"
    )
    print(f"Gaps detected: {len(record.gaps)}")
    print(f"Overlaps detected: {len(record.overlaps)}")
    if record.warnings:
        print("Warnings:")
        for w in record.warnings:
            print(f"  - {w}")

    if args.target_title:
        relevance = engine.score_relevance(
            record,
            target_title=args.target_title,
            job_keywords=args.job_keywords,
            job_id=args.job_id,
        )
        print(f"\nRelevance vs '{args.target_title}': {relevance.overall_relevance_score}")
        print(f"Relevant experience: {relevance.relevant_experience_years} years")

    print(f"\nOutput written to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
