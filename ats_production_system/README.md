# ATS Production System

**Day 20 deliverable — Zecpath AI Job Portal**
**Objective:** Validate the ATS AI module as complete and production-grade.

## Headline finding from this review

Before this module existed, the Day 12–14 "real" semantic matching and
scoring engines had never actually been exercised end-to-end. The Day 17
test harness's `pipeline_adapter.py` guesses at import paths
(`ats_scoring_engine.scorer.score_candidate`) that don't exist in the
real modules, so every test run silently fell through to a
keyword-overlap fallback scorer. The reported 87.5% accuracy figure was
therefore a measurement of that fallback, not the real pipeline.

A second issue, found while wiring the real engines together: their
output field names don't match what `ats_scoring_engine`'s loaders
expect (e.g. Semantic Matching emits `overall_score`, the loader wants
`similarity_score`; Education/Certification extraction has no
alignment-score concept against a job at all).

This module (`ats_production_system`) is the fix: a verified
end-to-end orchestrator (`pipeline.py`) plus a field-mapping adapter
layer (`component_adapters.py`) that makes the real Day 5–18 engines
talk to each other correctly, an API service (`api.py`) exposing the
core Day 16 spec, and an integration test suite that specifically
guards against this class of silent-fallback regression.

## What it does

`ATSPipeline` runs one candidate resume through the full chain and
produces an explainable, weighted final score:

```
resume file
  -> Day 5  ResumeExtractionEngine       (extract + clean)
  -> Day 18 PerformanceOptimizationEngine.repair_noisy_text (best-effort)
  -> Day 8  SectionClassifierEngine       (segment into sections)
  -> Day 9  SkillExtractionEngine          (candidate skills)
  -> Day 10 ExperienceParsingEngine        (timeline + job-relevance score)
  -> Day 11 EducationCertificationExtractor (degrees/certs)

job file
  -> Day 6  JDParsingEngine (required/preferred skills, experience & education requirements)

(resume outputs, job output)
  -> Day 12 SemanticMatchingEngine
  -> component_adapters.py   (Day 20 field-mapping fix)
  -> Day 13 ATSScoringEngine  (explainable final score, 4 weighted components)
  -> Day 14 CandidateRankingEngine (per-job ranking: Shortlist / Review / Auto-Reject)
  -> Day 15 FairnessBiasEngine      (pool-level score normalization + bias audit)
  -> Day 20 PipelineResultStore     (Day 7 metadata envelope)
```

Every stage is wrapped so a missing/failing upstream package degrades
just that component (logged as a warning, scored via renormalized
weights) instead of crashing the run — the same two-pass fallback
discipline used throughout the project.

## Project layout

```
ats_production_system/
├── __init__.py
├── pipeline.py             # ATSPipeline orchestrator
├── component_adapters.py   # Day 20 field-mapping fixes between engines
├── storage.py               # Day 7 envelope-based run/batch persistence
├── api.py                    # FastAPI service (core Day 16 spec)
├── cli.py                     # batch/demo CLI
├── Dockerfile
├── requirements.txt
├── .github/workflows/tests.yml
└── tests/
    ├── test_integration.py    # 12 end-to-end tests against the real pipeline
    ├── run_tests.py             # runs the suite + writes a timestamped log
    └── logs/                      # generated test-run logs (deliverable)

demo_datasets/                # Day 20 deliverable: demo resumes + job postings
├── resumes/                    # 6 synthetic .docx resumes, varied fit
└── job_descriptions/            # 2 job postings (Senior Backend Engineer, Junior Data Analyst)
```

## Requirements

```
pip install -r requirements.txt -r ats_production_system/requirements.txt
```

## Usage

### CLI (batch demo run)

```
python -m ats_production_system.cli \
    --resumes demo_datasets/resumes \
    --job demo_datasets/job_descriptions/senior_backend_engineer.txt \
    --job-id job_senior_backend \
    --output-dir outputs
```

### API service

```
uvicorn ats_production_system.api:app --host 0.0.0.0 --port 8000
```

Core endpoints (see `ats_api_design/openapi.yaml` for the full target spec):
- `POST /resumes` — upload a resume (.pdf/.docx)
- `POST /job-descriptions` — upload/submit a job posting
- `POST /matches` — score one resume against one job
- `GET /matches/{matchId}` — retrieve a stored match result
- `GET /job-descriptions/{jobDescriptionId}/shortlist` — ranked shortlist across all uploaded resumes
- `GET /health` — engine availability check

### Docker

```
docker build -f ats_production_system/Dockerfile -t zecpath-ats-ai:1.0.0 .
docker run -p 8000:8000 zecpath-ats-ai:1.0.0
```

## Running tests

```
python ats_production_system/tests/run_tests.py
```

12 tests cover: engine availability, job-skill extraction, full
component wiring for a strong-fit candidate, score discrimination
between strong/weak candidates, score bounds, a regression guard for
the exact field-mapping bug this review found, batch ranking + fairness
auditing, per-job ranking differences, graceful degradation when an
engine is unavailable, and unsupported-file-type error handling.

## Known limitations (explicit, not hidden)

- **Embedding engine**: this sandbox has no outbound access to Hugging
  Face, so `SemanticMatchingEngine` falls back to the offline hashing
  embedder. Semantic-similarity scores (and the `match_band`
  classification, which was calibrated for real embeddings) are
  therefore weaker signal than they would be in an environment with
  model access — every run logs which embedder was actually used.
- **API scope**: `api.py` implements the synchronous core of the Day 16
  spec, not every path in `openapi.yaml` (no async task polling via
  `/tasks/{id}`, no PATCH/update endpoints, no PDF/CSV shortlist
  export).
- **Persistence**: uploads and match records live in-process
  (dictionaries) in `api.py` for this reference implementation — a real
  deployment needs a database and file storage, not process memory.
- **No auth**: the API has no authentication/authorization layer.
- **Two remaining PRD microservices** (Interview Intelligence, Behavior
  Analysis) are out of scope for this module — this is the **ATS AI**
  service only, per the PRD's five-service split.

See the Day 20 Final ATS Evaluation Report for the full production-
readiness checklist.
