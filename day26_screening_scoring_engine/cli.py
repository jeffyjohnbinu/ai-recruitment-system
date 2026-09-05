"""
cli.py
------
Day 26 deliverable — Zecpath AI Job Portal

Command-line interface for the Screening Scoring Engine.

Run with:
    python -m day26_screening_scoring_engine.cli \\
        --candidate cand_001 \\
        --job job_backend_01 \\
        --session sess_001 \\
        --role software_engineer \\
        --answers answers.json

Where answers.json is a JSON file with shape:
    {
      "question_id": "Q-SE-006",
      "raw_answer": "I have 5 years of experience in Python and Django...",
      "expected_answer_type": "number",
      "category": "experience",
      "scoring_weight": 5,
      "is_mandatory": true,
      "extracted": {"years_experience": 5.0},
      "intent_label": "answer",
      "completeness": 0.9,
      "warnings": []
    }

Or use --input-json to pass answers as a JSON string directly.

For a demo with built-in sample answers:
    python -m day26_screening_scoring_engine.cli --demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from utils.logger import get_logger

logger = get_logger("day26_screening_scoring_engine.cli")


# ---- sample data for --demo ---- #

_DEMO_ANSWERS: List[Tuple[str, Dict[str, Any]]] = [
    (
        "Q-SE-001",
        {
            "raw_answer": "Yes, this is a good time to talk.",
            "expected_answer_type": "boolean",
            "category": "introduction",
            "scoring_weight": 1,
            "is_mandatory": True,
            "extracted": {"boolean": True},
            "intent_label": "answer",
            "completeness": 1.0,
            "warnings": [],
        },
    ),
    (
        "Q-SE-002",
        {
            "raw_answer": "My name is Arjun Nair and I applied for the Software Engineer position.",
            "expected_answer_type": "text",
            "category": "introduction",
            "scoring_weight": 1,
            "is_mandatory": True,
            "extracted": {},
            "intent_label": "answer",
            "completeness": 1.0,
            "warnings": [],
        },
    ),
    (
        "Q-SE-003",
        {
            "raw_answer": "I hold a B.Tech in Computer Science from IIT Madras.",
            "expected_answer_type": "text",
            "category": "education",
            "scoring_weight": 3,
            "is_mandatory": True,
            "extracted": {},
            "intent_label": "answer",
            "completeness": 0.9,
            "warnings": [],
        },
    ),
    (
        "Q-SE-004",
        {
            "raw_answer": "Yes, I have a B.Tech in Computer Science.",
            "expected_answer_type": "boolean",
            "category": "education",
            "scoring_weight": 5,
            "is_mandatory": True,
            "extracted": {"boolean": True},
            "intent_label": "answer",
            "completeness": 1.0,
            "warnings": [],
        },
    ),
    (
        "Q-SE-005",
        {
            "raw_answer": "I have around 5 years of total professional experience.",
            "expected_answer_type": "number",
            "category": "experience",
            "scoring_weight": 4,
            "is_mandatory": True,
            "extracted": {"years_experience": 5.0},
            "intent_label": "answer",
            "completeness": 0.85,
            "warnings": ["Vague qualifier 'around' detected."],
        },
    ),
    (
        "Q-SE-006",
        {
            "raw_answer": "I have about 4 years of experience in software development specifically.",
            "expected_answer_type": "number",
            "category": "experience",
            "scoring_weight": 5,
            "is_mandatory": True,
            "extracted": {"years_experience": 4.0},
            "intent_label": "answer",
            "completeness": 0.9,
            "warnings": [],
        },
    ),
    (
        "Q-SE-009",
        {
            "raw_answer": "I would rate my System Design skills as 4 out of 5.",
            "expected_answer_type": "number",
            "category": "skills",
            "scoring_weight": 4,
            "is_mandatory": True,
            "extracted": {},
            "intent_label": "answer",
            "completeness": 0.8,
            "warnings": [],
        },
    ),
    (
        "Q-SE-013",
        {
            "raw_answer": "I am currently based in Bengaluru.",
            "expected_answer_type": "text",
            "category": "location",
            "scoring_weight": 3,
            "is_mandatory": True,
            "extracted": {"city": "bengaluru"},
            "intent_label": "answer",
            "completeness": 1.0,
            "warnings": [],
        },
    ),
    (
        "Q-SE-014",
        {
            "raw_answer": "Yes, I am willing to relocate.",
            "expected_answer_type": "boolean",
            "category": "location",
            "scoring_weight": 4,
            "is_mandatory": True,
            "extracted": {"boolean": True},
            "intent_label": "answer",
            "completeness": 1.0,
            "warnings": [],
        },
    ),
    (
        "Q-SE-017",
        {
            "raw_answer": "I am looking for around 18 to 20 LPA.",
            "expected_answer_type": "number",
            "category": "salary",
            "scoring_weight": 4,
            "is_mandatory": True,
            "extracted": {"expected_salary_lakhs": 19.0},
            "intent_label": "answer",
            "completeness": 0.9,
            "warnings": ["Vague qualifier 'around' detected."],
        },
    ),
    (
        "Q-SE-019",
        {
            "raw_answer": "My current notice period is 30 days.",
            "expected_answer_type": "duration",
            "category": "notice_period",
            "scoring_weight": 4,
            "is_mandatory": True,
            "extracted": {"notice_period_days": 30},
            "intent_label": "answer",
            "completeness": 1.0,
            "warnings": [],
        },
    ),
    (
        "Q-SE-021",
        {
            "raw_answer": "I can join within a month of receiving the offer.",
            "expected_answer_type": "date",
            "category": "notice_period",
            "scoring_weight": 3,
            "is_mandatory": True,
            "extracted": {},
            "intent_label": "answer",
            "completeness": 0.7,
            "warnings": [],
        },
    ),
    # --- Poor quality answers for comparison ---
    (
        "Q-SE-007",
        {
            "raw_answer": "maybe something like that I guess",
            "expected_answer_type": "text",
            "category": "experience",
            "scoring_weight": 2,
            "is_mandatory": True,
            "extracted": {},
            "intent_label": "answer",
            "completeness": 0.3,
            "warnings": ["Vague qualifiers detected."],
        },
    ),
    (
        "Q-SE-011",
        {
            "raw_answer": "",
            "expected_answer_type": "text",
            "category": "skills",
            "scoring_weight": 3,
            "is_mandatory": False,
            "extracted": {},
            "intent_label": "no_response",
            "completeness": 0.0,
            "warnings": ["No response received."],
        },
    ),
]


# ---- CLI logic ---- #


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Day 26 — Screening Scoring Engine CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--candidate",
        default="demo_candidate",
        help="Candidate ID",
    )
    p.add_argument(
        "--job",
        default="demo_job",
        help="Job ID",
    )
    p.add_argument(
        "--session",
        default="demo_session",
        help="Session ID",
    )
    p.add_argument(
        "--role",
        default="software_engineer",
        help="Role ID (matches hr_screening_question_bank)",
    )
    p.add_argument(
        "--answers",
        type=Path,
        help="Path to a JSON file containing a list of answer records.",
    )
    p.add_argument(
        "--input-json",
        help="JSON string (list of answer records) passed directly.",
    )
    p.add_argument(
        "--output",
        type=Path,
        help="Output path for the score JSON (default: print to stdout).",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="Run with built-in sample answers (no file needed).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-question breakdown.",
    )
    return p


def _load_answers(
    answers_arg: Path | None,
    input_json: str | None,
    demo: bool,
) -> List[Tuple[str, Dict[str, Any]]]:
    if demo:
        logger.info("Running demo with built-in sample answers.")
        return _DEMO_ANSWERS

    if input_json:
        try:
            data = json.loads(input_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Error: invalid JSON in --input-json: {exc}") from exc
    elif answers_arg:
        data = json.loads(answers_arg.read_text(encoding="utf-8"))
    else:
        raise SystemExit("Error: specify --answers, --input-json, or --demo.")

    if not isinstance(data, list):
        raise SystemExit("Error: answers must be a JSON list of answer records.")

    out: List[Tuple[str, Dict[str, Any]]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        qid = str(item.get("question_id") or "")
        if not qid:
            continue
        out.append((qid, dict(item)))
    return out


def _print_session(session: Any, verbose: bool) -> None:
    """Pretty-print a SessionScore to stdout."""
    print(f"\n{'='*60}")
    print(f"  SCREENING SCORE — {session.candidate_id} / {session.session_id}")
    print(f"{'='*60}")
    print(f"  Role          : {session.role_id}")
    print(f"  Job           : {session.job_id}")
    print(f"  Recommendation: {session.recommendation.upper()}")
    print(f"  Normalized Score: {session.normalized_score:.1%}")
    print()
    print(f"  Questions     : {session.answered_questions}/{session.total_questions} answered")
    print(f"  Mandatory unanswered: {session.mandatory_unanswered}")
    print(f"  Hard filters failed: {session.hard_filters_failed or 'none'}")
    print()
    print(f"  Dimension averages:")
    for dim, score in sorted(session.overall_dimension_scores.items()):
        # Use ASCII-only bar characters (Windows console safe).
        bar_filled = "#" * int(score * 20)
        bar_empty = "-" * (20 - int(score * 20))
        print(f"    {dim:<15} [{bar_filled}{bar_empty}] {score:.0%}")
    print()
    print(f"  Narrative:")
    for line in _wrap(session.narrative, 60):
        print(f"    {line}")
    if session.warnings:
        print()
        print(f"  Warnings:")
        for w in session.warnings:
            print(f"    ⚠ {w}")

    if verbose and session.breakdown:
        print()
        print(f"  Per-question breakdown:")
        print(f"  {'QID':<12} {'CAT':<15} {'W':<3} {'SCORE':<7} {'EXPLANATION'}")
        print(f"  {'-'*12} {'-'*15} {'-'*3} {'-'*7} {'-'*30}")
        for b in session.breakdown:
            cat = b.category or "unknown"
            wid = b.scoring_weight
            score = b.overall_score
            expl = (b.explanation or "")[:45]
            print(f"  {b.question_id:<12} {cat:<15} {wid:<3} {score:.0%}     {expl}")


def _wrap(text: str, width: int) -> List[str]:
    words = text.split()
    lines, current = [], []
    for word in words:
        if sum(len(w) for w in current) + len(current) + len(word) <= width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines or [""]


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Load answers.
    try:
        answers = _load_answers(args.answers, args.input_json, args.demo)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Error loading answers: {exc}") from exc

    if not answers:
        raise SystemExit("Error: no valid answer records found.")

    # Run scoring.
    try:
        from day26_screening_scoring_engine import ScreeningScoringEngine

        engine = ScreeningScoringEngine()
        session = engine.score_session(
            candidate_id=args.candidate,
            job_id=args.job,
            session_id=args.session,
            role_id=args.role,
            answers=answers,
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Error running scoring engine: {exc}") from exc

    # Output.
    if args.output:
        from day26_screening_scoring_engine import ScreeningScoringStore

        store = ScreeningScoringStore()
        path = store.save_session(session)
        print(f"✓ Score written to {path}", file=sys.stderr)
    else:
        _print_session(session, verbose=args.verbose)

    # Exit code.
    if session.recommendation == "reject":
        sys.exit(1)
    elif session.recommendation == "insufficient_data":
        sys.exit(2)


if __name__ == "__main__":
    main()
