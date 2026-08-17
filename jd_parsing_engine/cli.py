"""
Command-line entry point.

Usage:
    python -m jd_parsing_engine.cli <input_path> [--output-dir OUTPUT_DIR]

    <input_path> may be a single .txt/.md job description file, or a
    directory of them.

Examples:
    python -m jd_parsing_engine.cli jobs/backend_engineer.txt
    python -m jd_parsing_engine.cli jobs/ --output-dir extracted/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .parser import JDParsingEngine


def main():
    parser = argparse.ArgumentParser(description="Job Description Parsing System")
    parser.add_argument(
        "input_path", help="Path to a job description file (.txt/.md) or a directory of them"
    )
    parser.add_argument(
        "--output-dir", default="outputs", help="Where to write structured JSON + AI profiles"
    )
    args = parser.parse_args()

    engine = JDParsingEngine(output_dir=args.output_dir)
    path = Path(args.input_path)

    if path.is_dir():
        records = engine.process_directory(path)
    elif path.is_file():
        records = [engine.process_file(path)]
    else:
        print(f"Path not found: {path}", file=sys.stderr)
        sys.exit(1)

    success = sum(1 for r in records if r.status == "success")
    partial = sum(1 for r in records if r.status == "partial")
    failed = sum(1 for r in records if r.status == "failed")

    print(
        f"\nProcessed {len(records)} file(s): {success} success, {partial} partial, {failed} failed"
    )
    for r in records:
        marker = {"success": "OK", "partial": "PARTIAL", "failed": "FAILED"}[r.status]
        role = r.normalized_role or r.raw_title or "(role not detected)"
        print(
            f"  [{marker}] {Path(r.source_file).name} -> role: {role}, "
            f"skills: {len(r.required_skills)} required / {len(r.preferred_skills)} preferred, "
            f"exp: {r.min_experience_years}-{r.max_experience_years} yrs"
        )

    print(f"\nOutput written to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
