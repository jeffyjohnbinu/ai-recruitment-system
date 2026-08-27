"""
Command-line entry point.

Single-candidate mode:
    python -m ats_scoring_engine.cli \\
        --candidate-id C001 --job-id J001 \\
        --skill-json skill.json --experience-json exp.json \\
        --education-json edu.json --semantic-json sem.json \\
        [--role software_engineer] [--missing-data-mode renormalize] \\
        [--weights-config config/role_weight_profiles.json] \\
        [--output-dir outputs]

Batch mode (Candidate Score Generator, Day 13 deliverable #3) -- scores
every entry in a JSON manifest (see tests/fixtures/manifest_sample.json),
persists each record, and writes a ranked leaderboard.json:
    python -m ats_scoring_engine.cli --manifest candidates_manifest.json \\
        [--missing-data-mode renormalize] \\
        [--weights-config config/role_weight_profiles.json] \\
        [--output-dir outputs]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .generator import CandidateScoreGenerator
from .scoring_engine import ATSScoringEngine
from .storage import ResultStore, build_score_record
from .weights import WeightProfileRegistry


def _load_json(path: str | None):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"Warning: file not found, skipping: {p}", file=sys.stderr)
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _build_engine(args) -> ATSScoringEngine:
    registry = (
        WeightProfileRegistry.from_json_file(args.weights_config)
        if args.weights_config
        else WeightProfileRegistry()
    )
    return ATSScoringEngine(registry=registry, missing_data_mode=args.missing_data_mode)


def _run_single(args) -> None:
    if not args.candidate_id or not args.job_id:
        print(
            "Error: --candidate-id and --job-id are required in single-candidate mode "
            "(or pass --manifest for batch mode).",
            file=sys.stderr,
        )
        sys.exit(1)

    engine = _build_engine(args)
    result = engine.compute_score(
        candidate_id=args.candidate_id,
        job_id=args.job_id,
        skill_data=_load_json(args.skill_json),
        experience_data=_load_json(args.experience_json),
        education_data=_load_json(args.education_json),
        semantic_data=_load_json(args.semantic_json),
        role=args.role,
    )

    record = build_score_record(result)
    store = ResultStore(args.output_dir)
    out_path = store.save(record)

    print(f"Status: {record.status}")
    if record.final_score is not None:
        print(f"Final score: {record.final_score:.3f} (role profile: {record.role_profile})")
    if record.components_missing:
        print(f"Missing components: {', '.join(record.components_missing)}")
    print(f"\n{record.narrative_explanation}")
    print(f"\nOutput written to: {out_path}")


def _run_batch(args) -> None:
    engine = _build_engine(args)
    store = ResultStore(args.output_dir)
    generator = CandidateScoreGenerator(engine=engine, store=store)

    records = generator.generate_from_manifest(args.manifest)
    leaderboard = generator.build_leaderboard(records)

    leaderboard_path = Path(args.output_dir) / "leaderboard.json"
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_path.write_text(
        json.dumps(leaderboard, indent=2),
        encoding="utf-8",
    )

    print(f"Scored {len(records)} candidate(s) " f"from manifest: {args.manifest}\n")

    for entry in leaderboard:
        score_str = f"{entry['final_score']:.3f}" if entry["final_score"] is not None else "N/A"
        print(
            f"  #{entry['rank']:<3} "
            f"{entry['candidate_id']:<10} "
            f"score={score_str:<6} "
            f"status={entry['status']}"
        )

    print(f"\nIndividual records written to: " f"{Path(args.output_dir).resolve()}")
    print(f"Leaderboard written to: {leaderboard_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="ATS Scoring Engine (Day 13)")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to a JSON manifest for batch scoring (Candidate Score Generator mode). "
        "See tests/fixtures/manifest_sample.json for the format.",
    )
    parser.add_argument("--candidate-id", default=None, help="Required in single-candidate mode")
    parser.add_argument("--job-id", default=None, help="Required in single-candidate mode")
    parser.add_argument("--role", default=None, help="Role name used to select a weight profile")
    parser.add_argument("--skill-json", help="Path to Day 9 skill extraction JSON")
    parser.add_argument("--experience-json", help="Path to Day 10 experience parsing JSON")
    parser.add_argument("--education-json", help="Path to Day 11 education/certification JSON")
    parser.add_argument("--semantic-json", help="Path to Day 12 semantic matching JSON")
    parser.add_argument(
        "--weights-config", help="Optional path to a JSON file of role weight profiles"
    )
    parser.add_argument(
        "--missing-data-mode",
        default="renormalize",
        choices=["renormalize", "neutral_fill", "strict"],
    )
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    if args.manifest:
        _run_batch(args)
    else:
        _run_single(args)


if __name__ == "__main__":
    main()
