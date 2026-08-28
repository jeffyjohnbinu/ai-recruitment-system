"""
Command-line entry point.

Usage:
    python -m fairness_bias_engine.cli \
        --profiles profiles.json \
        --score-components components.json \
        [--role-id senior_backend_engineer] \
        [--demographics demographics.json] \
        [--shortlist shortlist.json] \
        [--output-dir outputs/] \
        [--max-keyword-share 0.35] \
        [--diminishing-threshold 8]

Input file formats:
    profiles.json:           list of raw candidate profile dicts
    components.json:         {candidate_id: {component: [raw_score, weight]}}
    demographics.json (opt): {dimension: {candidate_id: group_label}}
    shortlist.json (opt):    {candidate_id: true/false}
"""

from __future__ import annotations

import argparse
import json

from .engine import FairnessBiasEngine
from .storage import FairnessResultStore


def _load_json(path: str | None):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Fairness, Normalization & Bias Reduction Engine (Day 15)"
    )
    parser.add_argument("--profiles", required=True, help="Path to candidate profiles JSON (list)")
    parser.add_argument(
        "--score-components", required=True, help="Path to per-candidate score components JSON"
    )
    parser.add_argument(
        "--role-id", default=None, help="Identifier for the role/pool being processed"
    )
    parser.add_argument(
        "--demographics", default=None, help="Optional demographic group labels JSON"
    )
    parser.add_argument("--shortlist", default=None, help="Optional shortlist flags JSON")
    parser.add_argument(
        "--output-dir", default="outputs", help="Where to write the fairness report JSON"
    )
    parser.add_argument("--max-keyword-share", type=float, default=0.35)
    parser.add_argument("--diminishing-threshold", type=int, default=8)
    args = parser.parse_args()

    profiles = _load_json(args.profiles)
    raw_components = _load_json(args.score_components)
    # JSON can't store tuples; components arrive as [raw, weight] lists -> convert to tuples
    score_components = {
        cid: {name: tuple(vals) for name, vals in comps.items()}
        for cid, comps in raw_components.items()
    }
    demographics = _load_json(args.demographics)
    shortlist = _load_json(args.shortlist)

    engine = FairnessBiasEngine(
        max_keyword_share=args.max_keyword_share,
        diminishing_threshold=args.diminishing_threshold,
    )

    result = engine.process_candidate_pool(
        raw_profiles=profiles,
        score_components=score_components,
        role_id=args.role_id,
        demographic_groups=demographics,
        shortlist_flags=shortlist,
    )

    store = FairnessResultStore(args.output_dir)
    record = result.to_storage_record()
    filename = f"fairness_report_{args.role_id or 'unspecified_role'}.json"
    path = store.save(record, filename)

    print(
        f"\nProcessed {len(result.per_candidate)} candidate(s) for role: {args.role_id or '(none)'}"
    )
    for c in result.per_candidate:
        print(
            f"  {c.candidate_id}: adjusted_score={c.adjusted_final_score:.3f} "
            f"keyword_capped={c.keyword_balance.dampening_applied} "
            f"masked_fields={len(c.masking.removed_fields)}"
        )
    for report in result.bias_reports:
        flag = (
            "FLAGGED"
            if report.flagged
            else ("INSUFFICIENT DATA" if report.insufficient_data else "OK")
        )
        print(f"  Bias check [{report.dimension}]: {flag}")

    print(f"\nFairness report written to: {path}")


if __name__ == "__main__":
    main()
