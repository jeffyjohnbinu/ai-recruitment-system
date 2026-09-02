# eligibility_decision_engine

**Day 21 — Zecpath pipeline.** Standalone, additive package. Drop
`eligibility_decision_engine/` into repo root; no changes to existing
packages required.

## What it does

Decides which candidates qualify for AI screening calls, based on
ATS output + recruiter-defined per-job rules. Each candidate gets
tagged `eligible`, `review`, or `rejected`, with an auditable list of
which rule checks passed/failed.

## Install / run tests

```bash
cd eligibility_decision_engine
pip install pytest
pytest -v
```

## Quick use

```python
from eligibility_decision_engine import (
    load_ats_results, load_job_rules, evaluate_batch, build_envelope,
)

candidates = load_ats_results("ats_output.json")
rules = load_job_rules("job_rules.json")
results = evaluate_batch(candidates, rules)
envelope = build_envelope(results, source="ats_output.json")
```

See `examples/run_demo.py` for a runnable end-to-end example, and
`examples/sample_ats_results.json` / `examples/sample_job_rules.json`
for the expected input shapes.

## Rule configuration format

One rule object per job role (array under `"rules"`, or a single
object). Alias-tolerant: `min_score`/`cutoff_score` also accepted for
`min_ats_score`, etc — see `loaders.py::RULE_ALIASES` for full list.

```json
{
  "job_role": "backend_engineer",
  "min_ats_score": 75,
  "review_band": 10,
  "mandatory_skills": ["python"],
  "min_experience_years": 2,
  "max_experience_years": 8,
  "allowed_locations": ["Bengaluru", "Chennai"],
  "allow_remote": true,
  "required_availability": null
}
```

- `review_band`: score points below `min_ats_score` that still get
  `review` instead of outright `rejected` (a soft cutoff zone).
- `allowed_locations` empty = no location restriction.
- `allow_remote` + candidate `remote_ok` bypasses location check.

## Candidate eligibility result structure

```json
{
  "candidate_id": "C001",
  "job_role": "backend_engineer",
  "status": "eligible",
  "ats_score": 88,
  "reasons": [
    {"rule": "mandatory_skills", "passed": true, "detail": "all mandatory skills present"},
    {"rule": "experience_range", "passed": true, "detail": "4.0y within range"},
    {"rule": "location", "passed": true, "detail": "Bengaluru in allowed locations"},
    {"rule": "availability", "passed": true, "detail": "no availability constraint configured"},
    {"rule": "ats_score", "passed": true, "detail": "88.0 >= min 75.0"}
  ]
}
```

## Decision logic

1. Hard rules (mandatory skills, experience range, location/remote,
   availability) — any failure forces `rejected`.
2. Score vs `min_ats_score`:
   - `>= min` → passes
   - within `review_band` below min → soft fail, `review`
   - below the review floor → `rejected`
3. All checks pass → `eligible`.

## Metadata envelope

`build_envelope()` wraps results in a `{"metadata": {...}, "data":
[...]}` shape (record count, per-status counts, timestamp, engine
version) matching the pipeline's Day 7 envelope pattern. **Note:**
the exact Day 7 envelope module from the existing pipeline wasn't
available when this package was built — if your repo's Day 7
envelope has different field names, adjust `envelope.py` to match;
nothing else in the package depends on its exact shape.

## Files

```
eligibility_decision_engine/
├── eligibility_decision_engine/
│   ├── __init__.py       # public API
│   ├── schema.py         # CandidateInput, RuleConfig, CandidateEligibilityResult
│   ├── loaders.py        # alias-tolerant JSON loaders
│   ├── engine.py         # rule-based + score-based decision logic
│   └── envelope.py       # Day 7-style metadata envelope
├── examples/
│   ├── sample_ats_results.json
│   ├── sample_job_rules.json
│   └── run_demo.py
├── tests/
│   ├── test_schema.py
│   ├── test_loaders.py
│   ├── test_engine.py
│   └── test_envelope.py
└── README.md
```
