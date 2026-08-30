# ATS System Testing

**Day 17 deliverable — Zecpath AI Job Portal**

Validates ATS accuracy, reliability, and role adaptability by running a
curated set of resume/job-description pairs — spanning tech, non-tech,
fresher, and senior profiles — through the pipeline and comparing the
AI output against **manual review** (human-recruiter ground truth).
Produces precision/recall/F1 metrics, a segment-level accuracy
breakdown, a structured mismatch log, and a prioritized improvement
backlog.

## What it does

1. **Fixtures** (`fixtures.py`) — 8 hand-authored resume/JD pairs, 2 per
   segment (tech-senior, tech-fresher, non-tech-senior, non-tech-fresher),
   one designed to be a genuine match and one a genuine mismatch per
   segment. Each carries a `GroundTruth` block: the shortlist decision,
   recommendation, and expected matched skills a human reviewer would
   assign, plus reviewer notes explaining the judgment.
2. **Pipeline adapter** (`pipeline_adapter.py`) — the single seam between
   this test suite and the real ATS pipeline. It tries, in order: (a)
   the real Day 12 Semantic Matching Engine / Day 13 ATS Scoring Engine
   / Day 14 Candidate Ranking Engine, if importable in the current
   environment, (b) the repo's baseline `ats_engine.matcher` scaffold,
   (c) a minimal built-in keyword-overlap scorer. Whichever tier
   resolves is recorded on the output as `engine_used`, so the report is
   explicit about what was actually tested.
3. **Harness** (`harness.py`) — runs every fixture through the adapter
   and pairs each AI output with its ground truth.
4. **Metrics** (`metrics.py`) — computes a confusion matrix (TP/FP/TN/FN)
   on the binary shortlist decision, derives precision/recall/F1/accuracy
   overall and broken down by `role_type` and `seniority`, and produces a
   structured mismatch log (false positive / false negative /
   recommendation-only mismatch) with the human reviewer's notes attached.
5. **Backlog** (`backlog.py`) — turns mismatch patterns into a prioritized,
   traceable improvement backlog (P0/P1/P2), where every item lists the
   specific case IDs that motivated it — no opaque scoring, consistent
   with the project's "explainability over black-box scores" principle.
6. **Storage** (`storage.py`) — persists a full test run as JSON wrapped
   in the Day 7 metadata envelope (`schema_version`, `model_version`,
   `pipeline_version`, `request_id`, `generated_at`).

## Project layout

```
ats_system_testing/
├── __init__.py
├── fixtures.py           # 8 labeled resume/JD test cases + ground truth
├── pipeline_adapter.py    # alias-tolerant seam into the real ATS pipeline
├── harness.py              # runs fixtures through the adapter
├── metrics.py               # precision/recall/F1, segment breakdown, mismatches
├── backlog.py                 # mismatch patterns -> prioritized backlog
├── storage.py                   # Day 7 metadata-envelope JSON output
├── cli.py                         # command-line entry point
└── tests/
    ├── test_ats_system_testing.py  # pytest suite (19 tests)
    ├── run_tests.py                  # runs the suite + writes a timestamped log
    └── logs/                           # generated test-run logs (deliverable)
```

## Usage

### CLI

```
python -m ats_system_testing.cli --output-dir outputs
```

Prints a per-case comparison table, overall accuracy/precision/recall/F1,
the mismatch log, and the improvement backlog, then writes a structured
JSON result file.

### Python API

```python
from ats_system_testing import ATSTestHarness, compute_metrics
from ats_system_testing.backlog import build_backlog

harness = ATSTestHarness()
results = harness.run()
metrics = compute_metrics(results)
backlog = build_backlog(metrics)
```

## Running tests

```
python ats_system_testing/tests/run_tests.py
```

Runs the full pytest suite (19 tests) and writes a timestamped log to
`tests/logs/test_run_<UTC timestamp>.log`. Tests cover: fixture-set
sanity (segment coverage, unique IDs, match/mismatch balance per
segment), the fallback pipeline adapter, an end-to-end harness smoke run,
metrics math against synthetic cases with known precision/recall values,
and backlog-generation rules.

## Important calibration note

The `ADVANCE_THRESHOLD` / `HOLD_THRESHOLD` constants in
`pipeline_adapter.py` apply **only** to the last-resort fallback
keyword-overlap scorer used when the real Day 12-14 engines aren't
importable (as in a fresh sandbox with just this module). They are
calibrated against the score distribution observed on this fixture set
(genuine matches scored 0.29-0.54 raw overlap; genuine mismatches
scored 0.04-0.15) and are **not** the same as `ATS_MIN_MATCH_SCORE`
(0.65) in `config/settings.py`, which was tuned for the real semantic
pipeline. When the real engines are present, this adapter defers to
them entirely and these constants are unused.

## Current results (fallback scorer, this environment)

Because the sandbox this suite was authored in does not have Days
12-14's real engine packages installed, the run below reflects the
**fallback keyword-overlap scorer**, not the production semantic
pipeline. See `ATS_Testing_Report_Day17.docx/pdf` for the full writeup;
headline numbers:

| Metric | Value |
|---|---|
| Accuracy | 87.5% (7/8) |
| Precision | 100% |
| Recall | 75% |
| F1 | 85.7% |

One false negative (a strong-fit fresher candidate scored just under
the advance threshold) and one recommendation-level (not shortlist-level)
mismatch were found. See `improvement_backlog` in the JSON output, or
the report, for the prioritized fixes.

## Known limitations

- **Small fixture set (n=8).** Two cases per segment is enough to
  exercise both precision and recall failure modes, but is not
  statistically significant. The improvement backlog itself recommends
  expanding coverage before treating any single run as conclusive.
- **Fallback scorer is keyword-overlap only.** It cannot see synonyms or
  paraphrases (e.g. "led migration to microservices" vs. "microservices
  experience"), which is precisely the gap the real Day 12 semantic
  matcher is designed to close. Numbers in this README reflect the
  fallback, not the production pipeline.
- **English-only, single-format fixtures.** All fixtures are English,
  well-formatted text. Multilingual resumes, scanned/OCR'd resumes, and
  heavily templated formats are not yet covered by this suite (Day 5's
  extraction engine has separate tests for extraction-layer edge cases).
