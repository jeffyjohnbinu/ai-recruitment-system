"""
Command-line entry point.

Usage:
    python -m education_certification_extractor.cli <input_path> [--output-dir OUTPUT_DIR]

    <input_path> is a plain-text file containing cleaned resume text
    (e.g. the output of Day 5's ResumeExtractionEngine, or a
    tests/outputs/cleaned_text/*.clean.txt file). It may also be a
    directory of such .txt files.

Examples:
    python -m education_certification_extractor.cli resume.clean.txt
    python -m education_certification_extractor.cli cleaned_text/ --output-dir extracted/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .extractor import EducationCertificationExtractor


def main():
    parser = argparse.ArgumentParser(description="Education & Certification Parsing Engine")
    parser.add_argument(
        "input_path", help="Path to a cleaned resume .txt file, or a directory of such files"
    )
    parser.add_argument(
        "--output-dir", default="outputs", help="Where to write the structured JSON profile"
    )
    args = parser.parse_args()

    engine = EducationCertificationExtractor(output_dir=args.output_dir)
    path = Path(args.input_path)

    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() == ".txt")
    elif path.is_file():
        files = [path]
    else:
        print(f"Path not found: {path}", file=sys.stderr)
        sys.exit(1)

    records = []
    for file in files:
        text = file.read_text(encoding="utf-8")
        records.append(engine.extract_from_text(text, source_file=str(file)))

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
            f"{r.degrees_found} degree(s), {r.certifications_found} certification(s), "
            f"highest_degree={r.highest_degree or 'n/a'}"
        )

    print(f"\nOutput written to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
