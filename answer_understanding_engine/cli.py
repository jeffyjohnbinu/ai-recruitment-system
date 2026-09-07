"""
answer_understanding_engine/cli.py
------------------------------------
Simple command-line entry point for testing the Answer Understanding Engine.

Run with:
    python -m answer_understanding_engine.cli <answer> --question-id Q-SE-006 --expected-slot years_experience

Expected output (example):
    {
        "question_id": "Q-SE-006",
        "raw_answer": "I have 5 years of experience in Python and Django.",
        "intent": {
            "label": "answer",
            "confidence": 0.8,
            "rationale": "Treating reply as an on-topic answer."
        },
        "slots": [
            {
                "name": "years_experience",
                "value": 5.0,
                "raw_span": "5 years",
                "confidence": 0.9
            },
            {
                "name": "skills",
                "value": ["python", "django"],
                "raw_span": "I have 5 years of experience in Python and Django.",
                "confidence": 0.85
            }
        ],
        "extracted": {
            "years_experience": 5.0,
            "skills": ["python", "django"]
        },
        "is_off_topic": false,
        "is_vague": false,
        "is_missing": false,
        "completeness": 1.0
    }
"""

import argparse
import json
import sys
from pathlib import Path

from utils.logger import get_logger

from .engine import AnswerUnderstandingEngine

logger = get_logger("answer_understanding_engine.cli")


def main():
    parser = argparse.ArgumentParser(
        description="Process a candidate screening answer with intent understanding."
    )
    parser.add_argument(
        "answer",
        help="The raw answer text from the candidate.",
    )
    parser.add_argument(
        "--question-id",
        help="The ID of the screening question being answered.",
    )
    parser.add_argument(
        "--expected-slot",
        help="The slot the question is intended to extract (e.g., years_experience).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as pretty-printed JSON.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level.",
    )

    args = parser.parse_args()

    # Setup logging
    from utils.logger import get_logger

    logger = get_logger("answer_understanding_engine.cli")
    logger.setLevel(args.log_level)

    engine = AnswerUnderstandingEngine()
    result = engine.understand(
        answer=args.answer,
        question_id=args.question_id or "",
        expected_slot=args.expected_slot,
    )

    output = result.to_dict() if args.json else str(result)
    print(output)

    if args.json:
        sys.exit(0)


if __name__ == "__main__":
    main()
