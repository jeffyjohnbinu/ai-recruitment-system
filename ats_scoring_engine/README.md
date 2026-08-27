# ATS Scoring Engine

**Day 13 deliverable — Zecpath AI Job Portal**

Combines the outputs of Day 9 (skill extraction), Day 10 (experience
relevance), Day 11 (education & certification), and Day 12 (semantic
matching) into a single, transparent, explainable candidate score, using a
configurable, per-role weight system.

## What it does

1. **Extracts** a normalized 0-1 score for each of the four scoring
   components from whatever upstream JSON shape Day 9-12 happen to produce,
   using alias-tolerant field lookup (direct score fields first, then
   ratio/composite fallbacks) — consistent with the additive-only,
   schema-evolution-via-alias-tolerance convention used since Day 8.
2. **Weights** each component according to a `WeightProfile` resolved by
   job role (`software_engineer`, `data_scientist`, `sales`, `executive`,
   or `default`), loadable from/saveable to a JSON config file so recruiters
   can retune weights without touching code.
3. **Handles missing data** via one of three configurable modes:
   - `renormalize` (default) — drop missing components and redistribute
     their weight proportionally among the components that *are* available.
   - `neutral_fill` — fill a missing component with a neutral score
     (default `0.5`) at its full configured weight.
   - `strict` — refuse to score (`status="insufficient_data"`) once more
     than `max_missing_for_strict` components are missing.
4. **Produces an explainable output**: a per-component breakdown (raw
   score, original weight, applied weight, weighted contribution, notes),
   a recruiter-readable narrative sentence, and a plain-English explanation
   per component — not just a bare number.
5. **Stores** results as structured JSON carrying the Day 7 metadata
   envelope (`schema_version`, `model_version`, `pipeline_version`,
   `candidate_id`, `job_id`).
6. Ships with an **automated test suite** (34 tests) and a runner that
   writes timestamped log files, matching the Day 5-12 delivery pattern.

## Scoring formula

```
final_score = Σ (component_raw_score × component_applied_weight)
```

for `component ∈ {skill_match, experience_relevance, education_alignment,
semantic_similarity}`, clamped to `[0, 1]`. `component_applied_weight`
equals the role profile's configured weight when all data is present, and
is adjusted per the missing-data mode above otherwise.

## Project layout

```
ats_scoring_engine/
├── __init__.py
├── loaders.py            # alias-tolerant extraction from Day 9-12 output
├── weights.py             # WeightProfile + WeightProfileRegistry (per-role, JSON-loadable)
├── scoring_engine.py       # ATSScoringEngine — the core weighted formula + missing-data logic
├── explainability.py        # narrative + per-component explanation text
├── storage.py                 # ScoreRecord (Day 7 metadata envelope) + ResultStore (JSON output)
├── generator.py                 # CandidateScoreGenerator — batch score generation + leaderboard
├── cli.py                         # command-line entry point (single-candidate + batch/manifest modes)
├── config/
│   └── role_weight_profiles.json  # example weight-profile config (default set, editable)
└── tests/
    ├── test_scoring_engine.py       # pytest suite for the formula/weights/explainability (34 tests)
    ├── test_generator.py              # pytest suite for the batch generator (13 tests)
    ├── run_tests.py                     # runs both suites + writes a timestamped log
    ├── fixtures/                          # sample Day 9/10/11/12-shaped JSON used as test data
    │   ├── manifest_sample.json             # example batch manifest (3 candidates, 1 job)
    │   └── candidates/                        # per-candidate JSON referenced by the manifest
    └── logs/                                    # generated test-run logs (deliverable)
```

## Deliverables at a glance

| Brief deliverable | Standalone artifact |
|---|---|
| ATS scoring engine | `scoring_engine.py` → `ATSScoringEngine` |
| Configurable weight system | `weights.py` → `WeightProfile` / `WeightProfileRegistry` (+ `config/role_weight_profiles.json`) |
| Candidate score generator | `generator.py` → `CandidateScoreGenerator` (+ `python -m ats_scoring_engine.cli --manifest ...`) |

## Requirements

No third-party runtime dependencies — the scoring formula, weight system,
and explainability layer are pure Python and consume plain dict/JSON output
from upstream modules.

```
pip install pytest --break-system-packages
```

## Usage

### Python API

```python
from ats_scoring_engine import ATSScoringEngine

engine = ATSScoringEngine(missing_data_mode="renormalize")

result = engine.compute_score(
    candidate_id="C001",
    job_id="J001",
    skill_data=skill_extraction_output,        # Day 9 output dict
    experience_data=experience_parsing_output, # Day 10 output dict
    education_data=education_output,           # Day 11 output dict
    semantic_data=semantic_matching_output,    # Day 12 output dict
    role="software_engineer",
)

print(result.final_score)          # 0.865
print(result.status)               # "scored" | "insufficient_data"
print(result.components_missing)   # []
```

Wrap with the Day 7 metadata envelope and persist:

```python
from ats_scoring_engine import ResultStore, build_score_record

record = build_score_record(result)
ResultStore("outputs").save(record)
```

Custom / per-role weights, and bypassing extraction with already-known
scores (useful for pipeline integration or testing):

```python
from ats_scoring_engine import WeightProfile, WeightProfileRegistry

registry = WeightProfileRegistry()
registry.register(WeightProfile("recruiter", {
    "skill_match": 0.3, "experience_relevance": 0.3,
    "education_alignment": 0.2, "semantic_similarity": 0.2,
}))

engine = ATSScoringEngine(registry=registry)
result = engine.compute_score(
    candidate_id="C002", job_id="J002", role="recruiter",
    score_overrides={"skill_match": 0.9, "experience_relevance": 0.8,
                      "education_alignment": 0.7, "semantic_similarity": 0.6},
)
```

### Candidate Score Generator (batch scoring + leaderboard)

`CandidateScoreGenerator` is the deliverable for actually *generating*
scores at volume — one candidate, a batch of candidates in memory, or an
entire JSON manifest of candidate/job pairs (each with inline data or
file paths to Day 9-12 output) — persisting every record and producing a
ranked leaderboard.

```python
from ats_scoring_engine import CandidateScoreGenerator, ResultStore

generator = CandidateScoreGenerator(store=ResultStore("outputs"))

# Batch, in memory:
records = generator.generate_batch(list_of_candidate_score_requests)

# Batch, from a manifest file:
records = generator.generate_from_manifest("candidates_manifest.json")

# Straight to a ranked leaderboard (best candidate first; anyone the
# engine couldn't score sorts to the bottom rather than being dropped):
leaderboard = generator.build_leaderboard(records)
```

Manifest format — a JSON array of entries, each supplying component data
either inline or as a file path resolved relative to the manifest file's
own directory (see `tests/fixtures/manifest_sample.json`):

```json
[
  {
    "candidate_id": "C001",
    "job_id": "J001",
    "role": "software_engineer",
    "skill_json": "candidates/C001/skill.json",
    "experience_json": "candidates/C001/experience.json",
    "education_json": "candidates/C001/education.json",
    "semantic_json": "candidates/C001/semantic.json"
  }
]
```

A component may be omitted entirely (see `C003` in the sample manifest,
which only has skill data) — the engine's configured missing-data mode
applies per candidate, same as single-candidate mode.

### CLI

Single-candidate mode:
```
python -m ats_scoring_engine.cli \
    --candidate-id C001 --job-id J001 --role software_engineer \
    --skill-json day9_output.json --experience-json day10_output.json \
    --education-json day11_output.json --semantic-json day12_output.json \
    --output-dir outputs
```

Batch / generator mode:
```
python -m ats_scoring_engine.cli --manifest candidates_manifest.json --output-dir outputs
```
Writes one JSON record per candidate/job pair plus a ranked
`outputs/leaderboard.json`, and prints a rank-ordered summary to stdout.

Any of the four `--*-json` flags (single mode) may be omitted; the engine
falls back to its configured missing-data mode. Pass `--weights-config
config/role_weight_profiles.json` (either mode) to load custom role weight
profiles.

### Sample output

```json
{
  "schema_version": "1.0.0",
  "model_version": "ats-scoring-engine-1.0.0",
  "pipeline_version": "zecpath-day13",
  "candidate_id": "C001",
  "job_id": "J001",
  "role_profile": "software_engineer",
  "status": "scored",
  "final_score": 0.8652,
  "components": [
    { "name": "skill_match", "raw_score": 0.833, "weight_applied": 0.4, "weighted_contribution": 0.3332 },
    { "name": "experience_relevance", "raw_score": 0.9, "weight_applied": 0.3, "weighted_contribution": 0.27 },
    { "name": "education_alignment", "raw_score": 1.0, "weight_applied": 0.1, "weighted_contribution": 0.1 },
    { "name": "semantic_similarity", "raw_score": 0.81, "weight_applied": 0.2, "weighted_contribution": 0.162 }
  ],
  "narrative_explanation": "Candidate C001 scored 87% overall against job J001 (role profile: software_engineer), driven primarily by strong education alignment (100%), while semantic similarity was comparatively weaker (81%)."
}
```

## Running tests

```
python ats_scoring_engine/tests/run_tests.py
```

Writes a timestamped log to `tests/logs/test_run_<UTC timestamp>.log`. 47
tests across two files:

- `test_scoring_engine.py` (34 tests) — alias-tolerant extraction (direct
  fields, ratio/composite fallbacks, missing data) for all four
  components; weight-profile validation and role resolution (including
  JSON round-trip); the full scoring engine across all three missing-data
  modes plus the all-data and all-missing edge cases; narrative and
  per-component explanation text; metadata-envelope + persistence.
- `test_generator.py` (13 tests) — single/batch generation, manifest
  loading (inline data, file-path data, partial data, missing files),
  persistence via `ResultStore`, and leaderboard ranking (including
  `insufficient_data` candidates sorting last rather than being dropped).

## Integration note for Day 14

Day 14 (Candidate Ranking & Shortlisting Engine) currently consumes Day 12
similarity scores directly. This module's `CandidateScoreGenerator`
produces a combined, weight-adjustable, ranked leaderboard that supersedes
a raw Day 12 score as a ranking input — but per the additive-only
integration convention, Day 14 has not been modified to consume it
automatically. If you'd like ranking to run on top of the Day 13
leaderboard instead of (or alongside) raw Day 12 scores, that's a small,
explicit change to Day 14's loader — flag it and it can be wired in as its
own update.

## Known limitations

- **Field-name assumptions**: loaders try a curated list of plausible field
  names/paths per upstream module (see `loaders.py`). If Day 9-12 output
  schemas evolve to use entirely new, unlisted field names, the component
  will report as unavailable rather than silently guessing — extend the
  alias lists in `loaders.py` rather than changing the engine's behavior.
- **Boolean-composite education scoring** (`meets_minimum_education` /
  `field_match`) is a simple weighted heuristic (0.6 / 0.4 split, +0.1
  certification bonus) used only when Day 11 doesn't expose a direct
  numeric alignment score — tune the constants in `loaders.py` if Day 11's
  real output uses a different implicit weighting.
- **`neutral_fill` mode** does not renormalize weights, so a missing
  component filled with a neutral 0.5 can pull an otherwise strong or weak
  candidate toward the middle at that component's full configured weight —
  this is intentional (it signals "unknown," not "good" or "bad") but is
  worth knowing when picking a missing-data mode for a given use case.
