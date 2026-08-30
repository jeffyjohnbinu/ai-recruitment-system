"""
Command-line entry point.

Usage:
    python -m performance_optimization_engine.cli benchmark [--output-dir DIR]
    python -m performance_optimization_engine.cli refine-entities <input.json>
    python -m performance_optimization_engine.cli clean-noisy <input.txt>

Examples:
    python -m performance_optimization_engine.cli benchmark
    python -m performance_optimization_engine.cli refine-entities skills.json
    python -m performance_optimization_engine.cli clean-noisy resume.clean.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .optimizer import PerformanceOptimizationEngine


def _cmd_benchmark(args: argparse.Namespace) -> int:
    engine = PerformanceOptimizationEngine(output_dir=args.output_dir)
    report = engine.run_benchmark_and_save()
    print(json.dumps(report, indent=2))
    print(f"\nBenchmark report written under: {Path(args.output_dir).resolve()}")
    return 0


def _cmd_refine_entities(args: argparse.Namespace) -> int:
    path = Path(args.input_path)
    if not path.exists():
        print(f"Path not found: {path}", file=sys.stderr)
        return 1

    data = json.loads(path.read_text(encoding="utf-8"))
    engine = PerformanceOptimizationEngine(output_dir=args.output_dir)
    result = engine.refine_entities(data)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def _cmd_clean_noisy(args: argparse.Namespace) -> int:
    path = Path(args.input_path)
    if not path.exists():
        print(f"Path not found: {path}", file=sys.stderr)
        return 1

    text = path.read_text(encoding="utf-8")
    engine = PerformanceOptimizationEngine(output_dir=args.output_dir)
    cleaned, report = engine.repair_noisy_text(text)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}.repaired.txt"
    out_path.write_text(cleaned, encoding="utf-8")

    print(json.dumps(report.to_dict(), indent=2))
    print(f"\nRepaired text written to: {out_path.resolve()}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Performance Optimization & Tuning Engine")
    parser.add_argument(
        "--output-dir", default="outputs", help="Where to write reports/output files"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("benchmark", help="Run the full benchmark suite and save a report")

    refine_parser = subparsers.add_parser(
        "refine-entities", help="Refine a JSON list/dict of extracted entities"
    )
    refine_parser.add_argument("input_path", help="Path to a JSON file of entities")

    clean_parser = subparsers.add_parser(
        "clean-noisy", help="Apply additional noise repair to cleaned resume text"
    )
    clean_parser.add_argument("input_path", help="Path to a cleaned .txt resume file")

    args = parser.parse_args()

    if args.command == "benchmark":
        sys.exit(_cmd_benchmark(args))
    elif args.command == "refine-entities":
        sys.exit(_cmd_refine_entities(args))
    elif args.command == "clean-noisy":
        sys.exit(_cmd_clean_noisy(args))


if __name__ == "__main__":
    main()
