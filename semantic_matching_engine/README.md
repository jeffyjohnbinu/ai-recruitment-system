# Semantic Matching Engine

**Day 12 deliverable — Zecpath AI Job Portal**

Moves resume ↔ job matching beyond keyword overlap (`ats_engine/matcher.py`)
to embedding-based **semantic** similarity, scored separately across
Skills, Experience, and Projects and blended into one overall match score.

## What it does

1. **Embeds** resume and job-description text using whichever engine is
   available:
   - **`sentence-transformers`** (`all-MiniLM-L6-v2`) when installed and
     reachable — true semantic embeddings that understand paraphrase /
     synonym relationships ("led a team" ≈ "managed engineers").
   - **Offline hashing fallback** (scikit-learn `HashingVectorizer` + a
     small synonym-canonicalization pass) when the above isn't available —
     fully deterministic, no network access, no corpus-fitting step.
2. **Compares** three sections independently — Skills, Experience,
   Projects — via cosine similarity, then blends them into one overall
   score using configurable weights (default: 45% / 35% / 20%).
3. **Classifies** the overall score into `strong_match` / `possible_match`
   / `no_match` bands using tunable thresholds, with a grid-search tuner
   (`tune_match_threshold`) that finds the F1-maximizing cutoff against a
   labeled validation set.
4. **Adapts** inputs from every prior pipeline day automatically — plain
   `{"skills": ..., "experience": ..., "projects": ...}` dicts, Day 8
   section-classifier maps, Day 9 skill-record lists, Day 10
   experience-record lists, Day 6 `JobRequirementRecord`-style job dicts,
   or raw flat text as a last resort.
5. **Reports** matching accuracy (precision / recall / F1 / accuracy /
   confusion matrix), broken down by job type, as both JSON and Markdown.
6. **Stores** results as structured JSON following the Day 7 metadata
   envelope convention (`candidate_id`, `job_id`, `schema_version`).

## Project layout

```
semantic_matching_engine/
├── __init__.py
├── embeddings.py                # SentenceTransformerEmbedder + HashingBoWEmbedder (fallback)
├── similarity.py                 # cosine similarity + weighted section comparison
├── adapters.py                     # normalizes Day 6/8/9/10 output shapes into flat sections
├── thresholds.py                    # ThresholdConfig + grid-search tuner
├── matcher.py                        # SemanticMatchingEngine orchestrator
├── reports.py                         # matching accuracy report (precision/recall/F1/confusion matrix)
├── storage.py                          # SemanticMatchRecord + ResultStore (JSON output)
├── cli.py                               # command-line entry point
├── generate_accuracy_report.py            # runs the validation set end-to-end, writes the report
└── tests/
    ├── test_matching.py             # pytest suite (29 tests)
    ├── run_tests.py                   # runs the suite + writes a timestamped log
    ├── fixtures/fixtures.py             # resumes/JDs across 3 job types + labeled validation set
    └── logs/                              # generated test-run logs (deliverable)
```

## Requirements

```
pip install scikit-learn numpy pytest --break-system-packages
```

Optional, for real semantic embeddings instead of the offline fallback:
```
pip install sentence-transformers --break-system-packages
```
(first use downloads the `all-MiniLM-L6-v2` model from Hugging Face; if
that network call fails — e.g. in a restricted environment — the engine
automatically falls back to the offline hashing embedder instead of
crashing, and logs which engine it ended up using.)

## Usage

### Python API

```python
from semantic_matching_engine import SemanticMatchingEngine

engine = SemanticMatchingEngine(output_dir="outputs")  # engine="auto" by default

record = engine.match(
    candidate_id="C001",
    job_id="J001",
    resume_input={
        "skills": "Python, FastAPI, PostgreSQL, AWS",
        "experience": "Senior backend engineer, led migration to microservices...",
        "projects": "Built an ETL pipeline processing 10TB/day...",
    },
    job_input={  # Day 6 JobRequirementRecord shape also works directly
        "role": "Backend Engineer",
        "skills": "Python, FastAPI, PostgreSQL, AWS, Kubernetes",
        "experience": "Looking for someone who has led backend teams...",
    },
)

print(record.overall_score)   # 0.0 - 1.0
print(record.match_band)      # "strong_match" | "possible_match" | "no_match"
print(record.is_match)        # bool
print(record.section_scores)  # per-section similarity + weight breakdown
```

Batch matching across multiple job types:
```python
results = engine.match_many([
    ("C001", "J001", resume_1, job_1),
    ("C002", "J002", resume_2, job_2),
])
```

Tuning thresholds against labeled data:
```python
from semantic_matching_engine import tune_match_threshold

result = tune_match_threshold(scores=[0.9, 0.3, 0.6], labels=[True, False, True])
print(result.best_threshold, result.best_f1)
```

### CLI

```
python -m semantic_matching_engine.cli --resume resume.json --job job.json \
    --candidate-id C001 --job-id J001
```

### Generating the accuracy report deliverable

```
python -m semantic_matching_engine.generate_accuracy_report --output-dir outputs
```

This runs the built-in multi-job-type validation set (engineering, data
science, product) through the engine, tunes the match threshold against
the labeled ground truth, and writes:
- `outputs/structured/semantic_matches/matching_accuracy_report.json`
- `outputs/structured/semantic_matches/matching_accuracy_report.md`

### Output structure

Each match is written to:
- `outputs/structured/semantic_matches/<candidate_id>__<job_id>.json`

## Running tests

```
python semantic_matching_engine/tests/run_tests.py
```

This runs the full pytest suite (29 tests) and writes a timestamped log to
`tests/logs/test_run_<UTC timestamp>.log`. Coverage includes: embedding
engine selection/fallback, cosine similarity edge cases (identical,
orthogonal, zero vectors), section-weighted similarity (including
missing-section handling and custom weight normalization), every
cross-day input adapter shape, threshold classification/validation/tuning,
end-to-end matching through `SemanticMatchingEngine` (including a fully
empty-input failure path), JSON persistence, and the accuracy report
generator (including its multi-job-type breakdown).

Tests run against the offline hashing fallback engine specifically, so the
suite is fast, deterministic, and requires no network access or model
download — matching the project's existing testing discipline (see
`resume_extraction_engine`'s OCR-optional pattern).

## Known limitations

- **Offline fallback quality**: the hashing-based fallback embedder is a
  bag-of-words approximation with a small hand-curated synonym list — it
  narrows but does not close the gap with real sentence embeddings. Two
  semantically identical but lexically dissimilar sentences (no shared or
  synonym-mapped tokens) will still score low. Install
  `sentence-transformers` for production-quality semantic matching.
- **Threshold portability**: `ThresholdConfig`'s default `match_threshold`
  (0.55) was chosen as a reasonable starting point but is **engine-specific**
  — sentence-transformer cosine scores and hashing-fallback cosine scores
  are not on directly comparable scales. Re-run
  `tune_match_threshold()` against a labeled sample whenever the embedding
  engine changes, and whenever meaningfully more labeled data accumulates
  (same "tune thresholds during testing" lesson learned in Day 9's
  confidence-scoring work).
- **No dedicated JD "projects" field**: Day 6's `JobRequirementRecord`
  doesn't have a first-class projects section, so the job-side adapter
  falls back to `responsibilities` or `role` text for that axis. This is
  a reasonable proxy but not a true apples-to-apples comparison against a
  resume's actual project descriptions.
- **Validation set size**: the bundled fixture validation set (6 pairs
  across 3 job types) is enough to exercise the tuning and reporting code
  paths end-to-end, but is too small to be a statistically meaningful
  accuracy benchmark. Treat `generate_accuracy_report.py`'s output as a
  worked example to run again against a larger, real labeled set.
