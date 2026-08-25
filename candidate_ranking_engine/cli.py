"""
Command-line entry point.

Usage:
    python -m candidate_ranking_engine.cli <input_dir> --job-id JOB_ID
        [--shortlist-threshold 0.75] [--review-threshold 0.50]
        [--top-n 10] [--output-dir OUTPUT_DIR]

    <input_dir> is a directory of Day 12 Semantic Matching Engine JSON
    output files (one per candidate/job pair). Records for other job_ids
    in the same directory are ignored.

Examples:
    python -m candidate_ranking_engine.cli outputs/matches --job-id job_123
    python -m candidate_ranking_engine.cli outputs/matches --job-id job_123 \
        --shortlist-threshold 0.8 --review-threshold 0.55 --top-n 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .ranker import CandidateRankingEngine, InvalidConfigError, NoCandidatesError, RankingConfig


def main():
    parser = argparse.ArgumentParser(description="Candidate Ranking & Shortlisting Engine")
    parser.add_argument("input_dir", help="Directory of Day 12 semantic matching JSON records")
    parser.add_argument("--job-id", required=True, help="Job ID to rank candidates against")
    parser.add_argument(
        "--shortlist-threshold",
        type=float,
        default=0.75,
        help="Score at/above which a candidate is auto-shortlisted (default: 0.75)",
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=0.50,
        help="Score at/above which a candidate goes to manual review (default: 0.50)",
    )
    parser.add_argument(
        "--top-n", type=int, default=10, help="Number of top candidates to highlight (default: 10)"
    )
    parser.add_argument(
        "--output-dir", default="outputs", help="Where to write structured JSON + recruiter CSV"
    )
    args = parser.parse_args()

    try:
        config = RankingConfig(
            shortlist_threshold=args.shortlist_threshold,
            review_threshold=args.review_threshold,
            top_n=args.top_n,
        )
    except InvalidConfigError as exc:
        print(f"Invalid configuration: {exc}", file=sys.stderr)
        sys.exit(1)

    engine = CandidateRankingEngine(config=config, output_dir=args.output_dir)
    input_dir = Path(args.input_dir)

    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        report = engine.rank_directory(input_dir, job_id=args.job_id)
    except NoCandidatesError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nRanking for job: {report.job_id}")
    print(
        f"Total candidates: {report.total_candidates}  |  "
        f"Shortlist: {report.shortlist_count}  |  "
        f"Review: {report.review_count}  |  "
        f"Auto-Reject: {report.auto_reject_count}"
    )
    print(f"\nTop {min(report.top_n, report.total_candidates)} candidate(s):")
    for c in report.top_candidates:
        name = f" ({c.candidate_name})" if c.candidate_name else ""
        print(f"  #{c.rank:<3} {c.candidate_id}{name:<20} {c.overall_score:.4f}  [{c.zone}]")

    print(f"\nOutput written to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
