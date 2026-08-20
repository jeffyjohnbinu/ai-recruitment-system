"""
Command-line entry point.

Usage:
    python -m skill_extraction_engine.cli <input_path> [--output-dir OUTPUT_DIR]

    <input_path> is a plain-text or cleaned-text file (e.g. Day 5's
    `outputs/cleaned_text/<name>.clean.txt`) or a directory of such files.

Examples:
    python -m skill_extraction_engine.cli cleaned_text/john_doe.clean.txt
    python -m skill_extraction_engine.cli cleaned_text/ --output-dir extracted_skills/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .extractor import SkillExtractionEngine


def main():
    parser = argparse.ArgumentParser(description="Skill Extraction Engine")
    parser.add_argument(
        "input_path", help="Path to a cleaned-text file (.txt) or a directory of such files"
    )
    parser.add_argument(
        "--output-dir", default="outputs", help="Where to write structured skill JSON"
    )
    args = parser.parse_args()

    engine = SkillExtractionEngine(output_dir=args.output_dir)
    path = Path(args.input_path)

    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() == ".txt")
        results = [engine.extract_from_file(f) for f in files]
    elif path.is_file():
        results = [engine.extract_from_file(path)]
    else:
        print(f"Path not found: {path}", file=sys.stderr)
        sys.exit(1)

    print(f"\nProcessed {len(results)} file(s):")
    for r in results:
        marker = {"success": "OK", "partial": "PARTIAL", "failed": "FAILED"}[r.status]
        top = ", ".join(s.skill for s in r.skills[:6])
        print(f"  [{marker}] {r.source_file} -> {r.total_skills_found} skills ({top}...)")

    print(f"\nOutput written to: {Path(args.output_dir).resolve() / 'structured'}")


if __name__ == "__main__":
    main()
