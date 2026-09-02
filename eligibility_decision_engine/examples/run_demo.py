"""Quick demo: load sample data, run engine, print envelope."""

import json
import sys
from pathlib import Path


def main():
    """Run the eligibility decision engine demo."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from eligibility_decision_engine import (
        build_envelope,
        evaluate_batch,
        load_ats_results,
        load_job_rules,
    )

    here = Path(__file__).resolve().parent

    candidates = load_ats_results(here / "sample_ats_results.json")
    rules = load_job_rules(here / "sample_job_rules.json")
    results = evaluate_batch(candidates, rules)
    envelope = build_envelope(results, source="sample_ats_results.json")

    print(json.dumps(envelope, indent=2))


if __name__ == "__main__":
    main()
