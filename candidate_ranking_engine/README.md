# Candidate Ranking & Shortlisting Engine

**Day 14 deliverable — Zecpath AI Job Portal**

Consumes the per-candidate/job similarity records produced by the **Day 12
Semantic Matching Engine** (or any compatible upstream output) and turns
them into a sorted, zoned, recruiter-friendly ranking: automatic
shortlisting, a manual-review zone, auto-rejection below threshold, and a
top-N candidate list.

## What it does

1. **Loads** candidate/job match records from a directory of JSON files,
   tolerating minor key-naming differences from upstream (`candidate_id` /
   `candidateId`, `overall_score` / `match_score` / `similarity_score`,
   etc.) so Day 12's output format never has to change to feed this module.
2. **Sorts** candidates for a job by overall match score, descending, with
   a deterministic tie-break (skills sub-score, then candidate ID) so
   re-running on the same input always produces the same order.
3. **Zones** every candidate into one of three bands using configurable
   thresholds:
   - **Shortlist** — score ≥ `shortlist_threshold` (default `0.75`)
   - **Review** — `review_threshold` ≤ score < `shortlist_threshold` (default `0.50`)
   - **Auto-Reject** — score < `review_threshold`
4. **Generates** a top-N list (default 10) for quick recruiter scanning,
   plus a plain-language reason string per candidate explaining why they
   landed in their zone.
5. **Stores** results as structured JSON (for pipeline/API consumption)
   and a recruiter-friendly CSV (opens directly in Excel/Sheets).
6. Ships with an **automated test suite** (24 tests) and a runner that
   writes timestamped log files.

## Project layout

```
candidate_ranking_engine/
├── __init__.py
├── ranker.py              # CandidateRankingEngine + RankingConfig (core orchestrator)
├── loader.py               # tolerant parsing of upstream (Day 12) match records
├── storage.py                # data models + ResultStore (JSON/CSV output)
├── cli.py                      # command-line entry point
└── tests/
    ├── test_ranking.py         # pytest suite (24 tests)
    ├── run_tests.py             # runs the suite + writes a timestamped log
    ├── sample_data/               # sample Day 12-style match records used as fixtures
    └── logs/                        # generated test-run logs (deliverable)
```

## Requirements

```
pip install pytest --break-system-packages
```

No other third-party dependencies — ranking and shortlisting logic is
pure Python (`dataclasses`, `csv`, `json`), consistent with keeping this
stage dependency-light and fast.

## Usage

### Python API

```python
from candidate_ranking_engine import CandidateRankingEngine, RankingConfig

engine = CandidateRankingEngine(
    config=RankingConfig(shortlist_threshold=0.75, review_threshold=0.50, top_n=10),
    output_dir="outputs",
)

# From a directory of Day 12 match-record JSON files:
report = engine.rank_directory("outputs/matches", job_id="job_100")

print(report.shortlist_count, report.review_count, report.auto_reject_count)
for c in report.top_candidates:
    print(c.rank, c.candidate_id, c.overall_score, c.zone)
```

Ranking an in-memory list (e.g. already loaded elsewhere in the pipeline):

```python
from candidate_ranking_engine.storage import CandidateMatchInput

candidates = [
    CandidateMatchInput(candidate_id="c1", job_id="job_100", overall_score=0.91),
    CandidateMatchInput(candidate_id="c2", job_id="job_100", overall_score=0.42),
]
report = engine.rank_job(candidates, job_id="job_100")
```

### CLI

```
python -m candidate_ranking_engine.cli outputs/matches --job-id job_100
python -m candidate_ranking_engine.cli outputs/matches --job-id job_100 \
    --shortlist-threshold 0.8 --review-threshold 0.55 --top-n 5 \
    --output-dir outputs/ranking
```

### Output structure

For each job ranked, two files are written:
- `outputs/structured/<job_id>_ranking.json` — full `RankingReport`
  (thresholds used, zone counts, top-N list, and the complete ranked list
  with per-candidate metadata envelope)
- `outputs/recruiter_csv/<job_id>_ranking.csv` — flat, recruiter-facing
  view: rank, candidate, score, band, zone, reason

## Running tests

```
python candidate_ranking_engine/tests/run_tests.py
```

This runs the full pytest suite and writes a timestamped log to
`tests/logs/test_run_<UTC timestamp>.log`. 24 tests cover: tolerant
loading of standard and alias-keyed match records, malformed-record
skipping, job-ID filtering, config validation (inverted/invalid
thresholds), descending sort with deterministic tie-breaking, zone
boundary correctness (inclusive at thresholds), custom-threshold
behavior, top-N truncation, reason-string content, metadata-envelope
presence, and full end-to-end directory → ranking → persisted-output
flow.

## Cross-day integration notes

- **Input contract**: any record exposing a candidate ID, job ID, and a
  numeric overall score (under any of the accepted key aliases) can be
  ranked — this module does not require changes to the Day 12 schema.
- **Metadata envelope**: every `RankedCandidate` and `RankingReport`
  carries `schema_version`, `model_version`, and `pipeline_version`,
  per the Day 7 convention.
- **Additive only**: new upstream fields (e.g. additional section scores)
  pass through into `section_scores` automatically; nothing here needs
  to change to consume them.

## Known limitations

- **Single-job ranking per call**: `rank_job` / `rank_directory` operate
  on one `job_id` at a time by design (recruiters rank candidates against
  a specific requisition). To rank multiple jobs from one directory, call
  the engine once per `job_id`.
- **No persistence of prior rankings**: each run is stateless — it does
  not diff against a previous ranking to highlight movement. That would
  be a natural extension for a future day (e.g. "candidate moved from
  Review to Shortlist since last run").
- **Thresholds are score-based, not quota-based**: the shortlist size is
  whatever clears `shortlist_threshold`, not a fixed "top 5" cutoff — use
  `top_n` alongside a lower `shortlist_threshold` if a fixed-size
  shortlist is what a recruiter wants instead.
