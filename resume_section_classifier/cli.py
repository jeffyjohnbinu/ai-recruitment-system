"""
Command-line entry point.

Usage:
    python -m resume_section_classifier.cli <input_path> [--output-dir OUTPUT_DIR]

    <input_path> may be a single .txt/.pdf/.docx file or a directory.

Examples:
    python -m resume_section_classifier.cli resumes/john_doe.txt
    python -m resume_section_classifier.cli resumes/ --output-dir sectioned/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .classifier import SectionClassifierEngine


def main():
    parser = argparse.ArgumentParser(description="Resume Section Classifier")
    parser.add_argument(
        "input_path", help="Path to a resume file (.txt/.pdf/.docx) or a directory of resumes"
    )
    parser.add_argument(
        "--output-dir", default="outputs", help="Where to write structured JSON + labeled text"
    )
    args = parser.parse_args()

    engine = SectionClassifierEngine(output_dir=args.output_dir)
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
        print(
            f"  [{marker}] {Path(r.source_file).name} -> "
            f"sections: {', '.join(r.sections_detected) or 'none detected'}"
        )

    print(f"\nOutput written to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
