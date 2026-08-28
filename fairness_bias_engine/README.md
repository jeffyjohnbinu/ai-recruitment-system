# Fairness, Normalization & Bias Reduction Engine

**Day 15 deliverable — Zecpath AI Job Portal**

Improves fairness, reduces bias, and standardizes resume evaluation. This
module sits between the Day 13 ATS Scoring Engine / Day 14 Candidate
Ranking Engine and any human decision-maker, and is additive-only: it
does not modify the schema or behavior of any prior day's module.

## What it does

1. **Normalizes** heterogeneous candidate profiles (assembled from Day
   5–11 outputs) into one canonical schema — standardized skills, a
   unified degree tier ladder, consistent experience-years units — so
   formatting differences between resumes never become an accidental
   scoring signal. (`normalizer.py`)
2. **Masks** non-essential personal attributes — name, gender, age,
   date of birth, marital status, nationality, religion, photo — before
   a profile reaches scoring, in the spirit of "blind hiring". Street
   addresses are trimmed to city/region only, gendered pronouns and
   honorifics in free text are neutralized, and institution names are
   reduced to a prestige-neutral placeholder by default. A reversible
   mapping is preserved for legitimate recruiter use but is never
   included in the payload handed to scoring or exported bias reports.
   (`masker.py`)
3. **Reduces keyword over-dependence** in the final score: applies a
   diminishing-returns curve to the keyword-match component so
   resume-stuffing has shrinking payoff, and caps the *share* of total
   weight the keyword component may hold (default 35%), redistributing
   any excess proportionally to the remaining Day 13 scoring components
   (semantic similarity, experience relevance, etc.). (`keyword_balancer.py`)
4. **Normalizes scores** across a candidate pool (typically: all
   applicants for one role) via z-score, percentile rank, and a
   median/IQR-based robust rescale, so a given score means the same
   thing across roles with different raw-score distributions.
   (`score_normalizer.py`)
5. **Audits bias indicators**: given optional, separately-supplied group
   labels (never used in scoring itself), computes per-group selection
   rates and applies the four-fifths (80%) adverse-impact screening
   rule, flagging when it's violated and reporting "insufficient data"
   rather than a false conclusion when group sizes are too small.
   (`bias_auditor.py`)

## Project layout

```
fairness_bias_engine/
├── __init__.py
├── normalizer.py          # ResumeNormalizer -> CanonicalCandidateProfile
├── masker.py                # AttributeMasker -> MaskingResult
├── keyword_balancer.py        # KeywordDependencyReducer -> KeywordBalanceReport
├── score_normalizer.py          # ScoreNormalizer -> NormalizedScoreRecord
├── bias_auditor.py                # BiasAuditor -> BiasAuditReport
├── engine.py                        # FairnessBiasEngine (orchestrator)
├── storage.py                        # FairnessResultStore (JSON output, Day 7 envelope)
├── cli.py                              # command-line entry point
└── tests/
    ├── test_fairness_bias_engine.py     # pytest suite (25 tests)
    ├── run_tests.py                       # runs the suite + writes a timestamped log
    ├── fixtures/                            # sample profiles / components / demographics
    └── logs/                                  # generated test-run logs (deliverable)
```

## Requirements

No new third-party dependencies — pure standard library (`statistics`,
`math`, `re`, `dataclasses`). Only `pytest` is needed for the test suite:

```
pip install pytest --break-system-packages
```

## Usage

### Python API

```python
from fairness_bias_engine import FairnessBiasEngine

engine = FairnessBiasEngine(max_keyword_share=0.35, diminishing_threshold=8)

result = engine.process_candidate_pool(
    raw_profiles=[...],              # list of raw candidate profile dicts
    score_components={               # {candidate_id: {component: (raw_score, weight)}}
        "cand_001": {
            "keyword_match": (0.9, 0.6),
            "semantic_similarity": (0.6, 0.4),
        },
    },
    role_id="senior_backend_engineer",
    demographic_groups={"gender": {"cand_001": "female"}},  # optional, audit-only
    shortlist_flags={"cand_001": True},                       # optional
)

for c in result.per_candidate:
    print(c.candidate_id, c.adjusted_final_score, c.masking.removed_fields)

for report in result.bias_reports:
    print(report.dimension, report.flagged, report.narrative)
```

### CLI

```
python -m fairness_bias_engine.cli \
    --profiles fairness_bias_engine/tests/fixtures/sample_profiles.json \
    --score-components fairness_bias_engine/tests/fixtures/sample_score_components.json \
    --demographics fairness_bias_engine/tests/fixtures/sample_demographics.json \
    --shortlist fairness_bias_engine/tests/fixtures/sample_shortlist.json \
    --role-id senior_backend_engineer \
    --output-dir outputs/
```

### Output structure

A single JSON file per role is written to `<output-dir>/fairness_report_<role_id>.json`,
containing masked profiles + canonical profiles, per-candidate keyword-balance
reports, pool-normalized scores, and any bias audit reports — tagged with the
standard `schema_version` / `model_version` / `pipeline_version` metadata
envelope (Day 7 convention).

## Running tests

```
python fairness_bias_engine/tests/run_tests.py
```

This runs the full pytest suite (25 tests) and writes a timestamped log to
`tests/logs/test_run_<UTC timestamp>.log`. Coverage:

- **Normalization**: degree-tier canonicalization, categorized-skills
  flattening + de-duplication, experience-year derivation fallbacks,
  graceful handling of unrecognized/missing fields.
- **Masking**: protected-field redaction, job-relevant fields left
  untouched, gendered-pronoun neutralization, address trimming,
  institution-name masking, reversible map never leaking into default output.
- **Keyword rebalancing**: weight capping + redistribution, no-op when
  already under cap, diminishing-returns curve, graceful handling when no
  keyword component is present.
- **Score normalization**: z-score/percentile correctness, single-candidate
  pool, zero-variance pool, empty pool.
- **Bias auditing**: four-fifths rule violation detection, no false flag
  when rates are close, insufficient-data reporting for missing/tiny groups.
- **End-to-end integration**: full pipeline from raw profiles to a storage
  record, confirming the reversible identity map never leaks out.

## Known limitations & design choices

- **No demographic inference.** This module never guesses a candidate's
  protected characteristics from resume content — that would itself be a
  bias vector. Group labels for the bias auditor must come from a
  separate, deliberate process (e.g. an anonymous post-decision compliance
  survey), and are used only for auditing, never for scoring.
- **Four-fifths rule is a screening heuristic**, not a legal or causal
  determination — the generated narrative says so explicitly, and flags
  are suppressed (reported as "insufficient data") when group sizes are
  too small to be meaningful (default threshold: 5 per group).
- **Institution-name masking is a default, not a mandate.** Some roles
  legitimately require verifying a specific accreditation; the
  `AttributeMasker(mask_institution_names=False)` flag exists for cases
  where that trade-off is deliberately made, and it is off by default
  everywhere else to reduce pedigree bias.
- **Keyword rebalancing operates on Day 13's component shape** —
  `{component_name: (raw_score, weight)}` — so it can be inserted into the
  pipeline without changing the Day 13 or Day 14 schemas.
