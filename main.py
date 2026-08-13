"""
main.py
--------
Demonstration entry point that wires the pipeline together:

    resume file -> parsers -> ats_engine -> screening_ai -> scoring

Run with:
    python main.py --resume data/raw/sample_resume.txt --job data/raw/sample_job.txt

For real PDF/DOCX resumes, parsers.resume_parser.parse_resume() is used
instead of the plain-text shortcut below.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ats_engine.matcher import match_resume_to_job
from scoring.scorer import compute_final_score
from utils.logger import get_logger

logger = get_logger("main")


def run_pipeline(resume_path: Path, job_path: Path) -> None:
    logger.info("Starting pipeline: resume=%s job=%s", resume_path, job_path)

    resume_text = resume_path.read_text(encoding="utf-8")
    job_text = job_path.read_text(encoding="utf-8")

    match_result = match_resume_to_job(resume_text, job_text)

    # NOTE: screening_ai.screener.screen_candidate() requires a live
    # OPENAI_API_KEY. Skipped here by default so the demo runs offline;
    # swap in a real call once your .env is configured.
    ai_recommendation = "advance" if match_result.is_shortlisted else "hold"

    final = compute_final_score(match_result.score, ai_recommendation)

    print("\n--- Pipeline Result ---")
    print(f"ATS match score : {match_result.score:.2f}")
    print(f"Matched keywords: {', '.join(match_result.matched_keywords) or '(none)'}")
    print(f"AI recommendation: {ai_recommendation}")
    print(f"Final score      : {final.final_score:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AI recruitment pipeline on one candidate.")
    parser.add_argument("--resume", type=Path, required=True, help="Path to a plain-text resume file.")
    parser.add_argument("--job", type=Path, required=True, help="Path to a plain-text job description file.")
    args = parser.parse_args()

    run_pipeline(args.resume, args.job)


if __name__ == "__main__":
    main()
