# Experience Parsing & Relevance Engine

**Day 10 deliverable — Zecpath AI Job Portal**

Understands a candidate's professional background from cleaned resume
text and computes how relevant that background is to a specific job.

## What it does

1. **Parses** the "Experience" section of cleaned resume text into
   structured entries: company, job title, start/end dates, and the
   bullet points describing each role. Tolerant of common separators
   (`—`, `--`, `-`, `,`) and date-range formats (`2021 – Present`,
   `Jan 2021 to Mar 2023`, `2019-Present`).
2. **Calculates total experience** by merging role date-ranges into a
   union of covered months, so overlapping roles (e.g. a full-time job
   plus freelance work) are never double-counted.
3. **Detects gaps** — stretches of time with no role coverage above a
   configurable threshold — and **overlaps** between concurrent roles,
   each attributed to the specific roles involved.
4. **Scores relevance** of each role (and the overall candidate) against
   a target job title + keyword list, combining:
   - Title-to-title similarity (token overlap + seniority-ladder
     closeness) — also exposed standalone as `title_similarity()` for
     general role-to-role / career-progression comparisons.
   - Keyword overlap between the role's title/bullets and the job's
     required-skill keywords (e.g. from the Day 6 JD Parsing Engine).
5. **Stores** results as structured JSON: a full `ExperienceRecord` per
   resume and an `ExperienceRelevanceRecord` per (candidate, job) pair.

## Project layout

```
experience_parsing_engine/
├── __init__.py
├── duration.py       # date-token parsing ("Jan 2021", "2021", "Present") + month math
├── parser.py           # ExperienceParser -> ExperienceEntry extraction
├── gaps.py               # total experience, gap detection, overlap detection
├── relevance.py            # title_similarity() + ExperienceRelevanceScorer
├── storage.py                # ExperienceRecord / ExperienceRelevanceRecord + ResultStore (JSON)
├── engine.py                    # ExperienceParsingEngine orchestrator
├── cli.py                          # command-line entry point
└── tests/
    ├── test_experience_engine.py    # pytest suite (28 tests)
    ├── run_tests.py                    # runs the suite + writes a timestamped log
    ├── sample_data/                      # sample resume text fixtures (clean/gap/overlap)
    └── logs/                                # generated test-run logs (deliverable)
```

## Requirements

No new third-party dependencies — pure-Python (stdlib `re`, `dataclasses`,
`datetime`). Requires `pytest>=7.0` for the test suite (already in
`resume_extraction_engine/requirements.txt`).

## Usage

### Python API

```python
from experience_parsing_engine import ExperienceParsingEngine

engine = ExperienceParsingEngine(output_dir="outputs")

# 1. Parse experience from cleaned resume text (Day 5 output feeds this directly)
record = engine.process_text(
    resume_text=cleaned_text,          # from resume_extraction_engine
    candidate_id="cand_0001",
    source_file="john_doe.docx",
)

print(record.status)                    # "success" | "partial" | "failed"
print(record.total_experience_years)    # e.g. 6.5
print(record.gaps)                      # [{...}]
print(record.overlaps)                  # [{...}]

# 2. Score relevance against a specific job (job_keywords typically come
#    from the Day 6 JD Parsing Engine's extracted required_skills list)
relevance = engine.score_relevance(
    record,
    target_title="Senior Backend Engineer",
    job_keywords=["python", "aws", "microservices", "rest"],
    job_id="job_0042",
)
print(relevance.overall_relevance_score)     # 0.0 - 1.0
print(relevance.relevant_experience_years)   # years spent in roles scoring >= 0.4
```

Standalone role-to-role similarity (no job context needed):

```python
from experience_parsing_engine import title_similarity

title_similarity("Staff Engineer", "Principal Engineer")   # -> partial similarity
title_similarity("Senior Backend Engineer", "Senior Backend Engineer")  # -> 1.0
```

### CLI

```bash
python -m experience_parsing_engine.cli resume.clean.txt --candidate-id cand_001

python -m experience_parsing_engine.cli resume.clean.txt --candidate-id cand_001 \
    --target-title "Senior Backend Engineer" \
    --job-keywords python aws microservices \
    --job-id job_001
```

### Output structure

- `outputs/experience/<name>.experience.json` — full `ExperienceRecord`
  (entries, total experience, gaps, overlaps, warnings)
- `outputs/relevance/<candidate_id>__<job_id>.relevance.json` — full
  `ExperienceRelevanceRecord` (per-role and overall relevance scores)

Both follow the pipeline's metadata-envelope conventions
(`candidate_id`, `job_id`, `schema_version`) established in the Day 7
storage design.

## Running tests

```bash
python experience_parsing_engine/tests/run_tests.py
```

28 tests cover: date-token parsing (year-only, month+year, "Present"),
role-header extraction (title/company/date-range across separator
styles), bullet attribution, total-experience union math, gap detection,
overlap detection (and confirming overlaps don't inflate totals),
title-similarity scoring (identical / adjacent-seniority / unrelated
titles), relevance scoring and keyword matching, and end-to-end engine
orchestration including JSON persistence.

## Known limitations & design notes

- **Role-header format**: extraction relies on a `Title <sep> Company
  (Start <range-sep> End)` line shape. This matches every fixture
  produced by the Day 5 engine's cleaner/normalizer, but a resume with
  wildly different formatting (e.g. dates on a separate line from the
  title) would need an additional line-pairing heuristic — flagged in
  `warnings` via `unmatched_line_count` rather than silently dropped.
- **Bug caught during build**: the initial "unrelated titles" test case
  assumed two titles with zero token overlap should score near 0, but
  the seniority-ladder component defaults both untagged titles to the
  same "mid" level, contributing a 0.3 floor. This is intentional (two
  unspecified-seniority roles shouldn't be treated as maximally
  distant), but the test was tightened to assert on that documented
  floor rather than an unqualified "low score."
- **Gap threshold** defaults to 1 month and is configurable via
  `ExperienceParsingEngine(gap_threshold_months=...)`.
- **Total experience** is computed on the *union* of date ranges, so a
  candidate with two simultaneous full-time-equivalent roles is not
  credited with 2x experience for the overlapping months — the overlap
  itself is still reported separately in `overlaps` for reviewer
  visibility.
