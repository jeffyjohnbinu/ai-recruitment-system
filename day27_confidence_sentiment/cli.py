"""
cli.py
------
Day 27 deliverable — Zecpath AI Job Portal

Command-line interface for the confidence & sentiment analysis engine.

Usage:
    python -m day27_confidence_sentiment.cli \\
        --candidate cand_001 \\
        --job job_backend_01 \\
        --session sess_001 \\
        --role software_engineer \\
        --answers q1,experience,"I have 5 years of Python." \\
        --answers q2,salary,"My expected is 15 LPA." \\
        --answers q3,skills,"I'm confident and excited."

    python -m day27_confidence_sentiment.cli \\
        --candidate cand_001 \\
        --job job_backend_01 \\
        --session sess_001 \\
        --role software_engineer \\
        --json-output

For full-featured test with sample data:
    python -m day27_confidence_sentiment.cli --demo
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Tuple

from .engine import ConfidenceSentimentEngine

# ---- Demo data ---- #

DEMO_ANSWERS: List[Tuple[str, Dict[str, Any]]] = [
    (
        "q1",
        {
            "raw_answer": "I have five years of experience in Python, Django, and React. I led a team of eight engineers on multiple projects.",
            "category": "experience",
        },
    ),
    (
        "q2",
        {
            "raw_answer": "My highest qualification is a B.Tech in Computer Science from VTU, graduated in 2019.",
            "category": "education",
        },
    ),
    (
        "q3",
        {
            "raw_answer": "I am based in Bangalore and am happy to relocate if needed.",
            "category": "location",
        },
    ),
    (
        "q4",
        {
            "raw_answer": "My expected salary is 18 LPA, and I am flexible to discuss based on the role.",
            "category": "salary",
        },
    ),
    (
        "q5",
        {
            "raw_answer": "Um, I think I can join within a month, maybe two weeks if we can negotiate the notice period.",
            "category": "notice_period",
        },
    ),
    (
        "q6",
        {
            "raw_answer": "I am excited and confident about this opportunity. I am passionate about building great software.",
            "category": "skills",
        },
    ),
]


def _print_session(result: Any) -> None:
    """Pretty-print a SessionConfidenceResult to stdout."""
    print("\n" + "=" * 60)
    print("  ZECPATH — Day 27: Confidence & Sentiment Signal Analysis")
    print("=" * 60)

    print(f"\n  Candidate : {result.candidate_id}")
    print(f"  Job       : {result.job_id}")
    print(f"  Session   : {result.session_id}")
    print(f"  Role      : {result.role_id}")
    print(f"  Answers   : {len(result.per_answer)}")

    print(f"\n  --- Session Aggregates ---")
    print(f"  Avg Confidence   : {result.session_avg_confidence:.0%}")
    print(f"  Avg Strength    : {result.session_avg_strength:.0%}")
    print(
        f"  Sentiment       : {result.session_sentiment_label} (polarity {result.session_sentiment_polarity:+.2f})"
    )
    print(f"  Hesitation Rate : {result.session_hesitation_rate:.1f} per answer")
    print(f"  Uncertainty Rate: {result.session_uncertainty_rate:.1f} per answer")
    print(
        f"  Avg Pace        : {result.avg_pace_wps:.1f} wps"
        if result.avg_pace_wps
        else "  Avg Pace        : N/A"
    )
    print(f"  Avg Filler Ratio: {result.avg_filler_ratio:.1%}")
    print(f"  Avg Length      : {result.avg_response_length:.0f} words")

    print(f"\n  --- Strength Distribution ---")
    for label, count in result.strength_distribution.items():
        if count > 0:
            print(f"    {label:15s}: {count}")

    if result.high_hesitation_answers:
        print(f"\n  ⚠ High Hesitation : {', '.join(result.high_hesitation_answers)}")
    if result.high_uncertainty_answers:
        print(f"\n  ⚠ High Uncertainty: {', '.join(result.high_uncertainty_answers)}")
    if result.low_confidence_answers:
        print(f"\n  ⚠ Low Confidence  : {', '.join(result.low_confidence_answers)}")
    if result.negative_sentiment_answers:
        print(f"\n  ⚠ Negative Sentiment: {', '.join(result.negative_sentiment_answers)}")
    if result.contradictions_detected:
        print(f"\n  ⚠ Contradictions : {len(result.contradictions_detected)} detected")

    print(f"\n  --- Per-Answer Breakdown ---")
    for ans in result.per_answer:
        si = ans.strength_indicator
        print(f"\n    Q{ans.question_id} ({ans.category or 'unknown'})")
        print(f"      Confidence  : {ans.overall_confidence_score:.0%}")
        print(f"      Strength    : {si.overall_strength:.0%} ({si.strength_label})")
        print(f"      Hesitations : {len(ans.hesitation_patterns)}")
        print(f"      Uncertainties: {len(ans.uncertainty_signals)}")
        print(
            f"      Sentiment   : {ans.sentiment.sentiment_label} ({ans.sentiment.polarity:+.2f})"
        )
        print(
            f"      Pace        : {ans.pace_metrics.pace_label} ({ans.pace_metrics.words_per_second:.1f} wps)"
            if ans.pace_metrics.words_per_second
            else f"      Pace        : {ans.pace_metrics.pace_label}"
        )

    if result.warnings:
        print(f"\n  --- Warnings ---")
        for w in result.warnings:
            print(f"    ⚠ {w}")

    print(f"\n  --- Narrative ---")
    print(f"  {result.narrative}")

    print(f"\n  Request ID: {result.request_id}")
    print(f"  Generated : {result.generated_at}")
    print("\n" + "=" * 60 + "\n")


def _parse_answers(raw: List[str]) -> List[Tuple[str, Dict[str, Any]]]:
    """Parse --answers flag values into (question_id, record) tuples.

    Format: question_id,category,raw_answer
    Example: q1,experience,I have 5 years of experience.
    """
    results: List[Tuple[str, Dict[str, Any]]] = []
    for item in raw:
        parts = item.split(",", 2)
        if len(parts) < 3:
            print(f"Warning: Skipping invalid --answers format: '{item}'", file=sys.stderr)
            continue
        qid, category, answer = parts
        results.append((qid, {"category": category, "raw_answer": answer}))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Day 27 confidence & sentiment analysis on interview answers."
    )
    parser.add_argument("--candidate", default="cand_demo", help="Candidate ID")
    parser.add_argument("--job", default="job_demo", help="Job ID")
    parser.add_argument("--session", default="sess_demo", help="Session ID")
    parser.add_argument("--role", default="software_engineer", help="Role ID")
    parser.add_argument(
        "--answers",
        action="append",
        metavar="QID,CATEGORY,ANSWER_TEXT",
        help="One or more answers in format: question_id,category,raw_answer_text",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Output results as JSON instead of formatted text.",
    )
    parser.add_argument("--demo", action="store_true", help="Run with built-in demo data.")
    parser.add_argument(
        "--use-llm", action="store_true", help="Enable LLM-enhanced scoring (stub)."
    )

    args = parser.parse_args()

    # Resolve answers.
    if args.demo:
        answers = DEMO_ANSWERS
        print("Using demo data (6 sample answers)...\n")
    elif args.answers:
        answers = _parse_answers(args.answers)
        if not answers:
            print("Error: No valid answers provided.", file=sys.stderr)
            sys.exit(1)
    else:
        print("Error: Provide --demo or --answers.", file=sys.stderr)
        sys.exit(1)

    # Run engine.
    engine = ConfidenceSentimentEngine(use_llm=args.use_llm)
    result = engine.score_session(
        candidate_id=args.candidate,
        job_id=args.job,
        session_id=args.session,
        role_id=args.role,
        answers=answers,
    )

    # Output.
    if args.json_output:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        _print_session(result)


if __name__ == "__main__":
    main()
