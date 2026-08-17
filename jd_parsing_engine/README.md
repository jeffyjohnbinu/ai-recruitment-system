# Job Description Parsing System

**Day 6 deliverable — Zecpath AI Job Portal**

Converts raw employer job descriptions (plain text pasted from a job board,
ATS export, or ticket) into a clean, normalized, structured **Job
Requirement Object** that downstream AI modules (`ats_engine` matching,
`screening_ai`, `interview_ai`) can consume reliably — mirroring the shape
of Day 5's `resume_extraction_engine` output so both sides of the pipeline
(resume ↔ JD) speak the same structured language.

## What it does

1. **Cleans** the raw JD text: strips control characters and stray
   symbols, collapses whitespace/blank-line noise, de-hyphenates
   line-wrapped words, and drops boilerplate that carries no requirement
   signal (EEO statements, "Apply now" footers, job-ID/posted-date lines,
   page numbers).
2. **Normalizes** structure: unifies bullet styles into a single `- `
   format and canonicalizes section headings (e.g. "What You'll Need" /
   "Qualifications" / "Must Haves" → `Requirements`) across the many ways
   job boards phrase the same section.
3. **Extracts** four structured fields:
   - **Role** — job title, normalized against a role/synonym table (e.g.
     "SWE II" → role `Software Engineer`, seniority `Mid-Level`), plus a
     detected seniority level (Intern → Executive).
   - **Skills** — required vs. preferred, normalized against a
     skill/synonym table (e.g. "JS", "Javascript", "ES6" all →
     `JavaScript`) so ATS matching isn't defeated by phrasing differences.
   - **Experience** — minimum/maximum years required, parsed from
     phrasings like "5+ years", "3-5 years", "minimum of 3 years".
   - **Education** — degree level (High School → PhD) and field(s) of
     study, scoped to text near the actual degree mention to avoid
     false positives (see *Known limitations*).
4. **Builds** a `JobRequirementRecord` — the full structured object,
   including cleaning stats, detected sections, and warnings for any
   field that could not be confidently extracted.
5. **Prepares AI-friendly JD profiles** — `to_ai_profile()` /
   `outputs/ai_profiles/*.json` — a trimmed view (role, seniority, skills,
   experience, education) with no extraction/metadata noise, ready to
   drop straight into an LLM prompt or the `ats_engine` matcher.
6. Ships with an **automated test suite** (29 tests) and a runner that
   writes timestamped log files, matching the Day 5 testing pattern.

## Project layout

```
jd_parsing_engine/
├── __init__.py
├── parser.py           # main orchestrator (JDParsingEngine)
├── cleaner.py           # JDTextCleaner: noise removal, heading/bullet normalization
├── synonyms.py            # skill & role alias tables + seniority keywords
├── extractors.py            # extract_role / extract_skills / extract_experience / extract_education
├── storage.py                  # JobRequirementRecord + JDResultStore (JSON output)
├── cli.py                        # command-line entry point
└── tests/
    ├── test_jd_parsing.py         # pytest suite (29 tests)
    ├── run_tests.py                 # runs the suite + writes a timestamped log
    ├── sample_jds/                    # 3 sample job descriptions used as fixtures
    ├── outputs/                         # structured JD output samples (deliverable)
    │   ├── structured/                    # full JobRequirementRecord JSON
    │   ├── cleaned_text/                    # plain cleaned text
    │   └── ai_profiles/                       # trimmed AI-ready JD profiles
    └── logs/                                    # generated test-run logs (deliverable)
```

## Requirements

No third-party dependencies for the core module — everything is
standard-library `re` / `dataclasses` / `pathlib`. Testing uses `pytest`
(already in the project's `requirements.txt`).

```
pip install pytest --break-system-packages
```

## Usage

### Python API

```python
from jd_parsing_engine import JDParsingEngine

engine = JDParsingEngine(output_dir="outputs")

# From a file
record = engine.process_file("jobs/backend_engineer.txt")

# Or from text already in memory (e.g. pasted from an ATS/job board)
record = engine.process_text(jd_text, source_name="backend_engineer")

print(record.status)              # "success" | "partial" | "failed"
print(record.normalized_role)     # "Backend Engineer"
print(record.required_skills)     # ["Python", "SQL", "Docker", ...]
print(record.min_experience_years, record.max_experience_years)  # 5.0, None
print(record.education_level)     # "Bachelor's"

# AI-ready trimmed profile for prompts / matching
ai_profile = record.to_ai_profile()
```

Batch processing a folder of `.txt`/`.md` postings:
```python
records = engine.process_directory("jobs/")
```

> **Note on PDF/DOCX job postings:** this module parses plain text.
> For a JD delivered as a PDF or Word file, first extract text with Day 5's
> `resume_extraction_engine.readers` (`PDFReader` / `DOCXReader` — they're
> file-format readers, not resume-specific), then pass the resulting text
> into `JDParsingEngine.process_text()`.

### CLI

```
python -m jd_parsing_engine.cli jobs/backend_engineer.txt
python -m jd_parsing_engine.cli jobs/ --output-dir extracted/
```

### Output structure

For each job description, three files are written:
- `outputs/structured/<name>.json` — full `JobRequirementRecord`
  (role, skills, experience, education, cleaning stats, warnings,
  cleaned text)
- `outputs/cleaned_text/<name>.clean.txt` — plain cleaned text only
- `outputs/ai_profiles/<name>.profile.json` — the trimmed AI-friendly
  profile (`to_ai_profile()`), ready to feed into `ats_engine` or an LLM
  prompt without extra parsing

## Job Requirement Object — field reference

| Field                  | Type            | Example                                  |
|-------------------------|-----------------|--------------------------------------------|
| `raw_title`              | `str \| None`   | `"Senior Backend Engineer"`                |
| `normalized_role`         | `str \| None`   | `"Backend Engineer"`                        |
| `seniority_level`           | `str \| None`   | `"Senior"`                                    |
| `required_skills`             | `list[str]`     | `["Python", "SQL", "Docker", "Kubernetes"]`     |
| `preferred_skills`              | `list[str]`     | `["Kafka", "Terraform"]`                          |
| `min_experience_years`             | `float \| None` | `5.0`                                               |
| `max_experience_years`               | `float \| None` | `None` (open-ended "5+ years")                        |
| `experience_mentions`                  | `list[str]`     | `["5+ years"]` (raw phrasing matched)                   |
| `education_level`                        | `str \| None`   | `"Bachelor's"`                                            |
| `education_fields`                         | `list[str]`     | `["Computer Science"]`                                      |
| `sections_detected`                          | `list[str]`     | `["Role Overview", "Requirements", ...]`                       |
| `warnings`                                     | `list[str]`     | e.g. `"No explicit years-of-experience requirement was detected."` |
| `status`                                         | `str`           | `"success"` \| `"partial"` \| `"failed"`                              |

`status` is `"partial"` (not `"failed"`) whenever text was extracted but
one or more fields (role, skills, experience, education) couldn't be
confidently detected — the record is still usable, just flagged for
review via `warnings`.

## Skill & role synonym normalization

`synonyms.py` holds two static alias tables:

- `SKILL_ALIASES` — canonical skill → list of aliases (abbreviations,
  casing variants, "X.js" forms, etc.), e.g. `"JavaScript": ["javascript",
  "js", "ecmascript", "es6"]`.
- `ROLE_ALIASES` — canonical role title → list of aliases/abbreviations,
  e.g. `"Software Engineer": ["swe", "sde", "software developer", ...]`.

Both are plain `dict[str, list[str]]` structures specifically so they're
easy to extend without touching extraction logic — add a new posting's
phrasing as an alias and normalization picks it up automatically. A third
table, `SENIORITY_KEYWORDS`, maps level names (Intern → Executive) to the
words that signal them in a title or opening lines.

## Running tests

```
python jd_parsing_engine/tests/run_tests.py
```

This runs the full pytest suite and writes a timestamped log to
`tests/logs/test_run_<UTC timestamp>.log`, in addition to printing the
result to stdout. 29 tests cover: end-to-end parsing of three fixture JDs
(clean/structured, experience-range + label-based title, and noisy/
minimal formatting), role normalization and seniority detection, required
vs. preferred skill separation and de-duplication, experience range/plus
parsing, education degree + field-of-study detection, boilerplate/noise
removal, error handling for missing or unsupported files, and isolated
cleaner/extractor unit behavior.

## Known limitations

- **Field-of-study scoping**: to avoid false positives (e.g. matching
  "Engineering" inside an unrelated phrase like "software engineering
  experience"), field-of-study keywords are only searched in a small
  text window right after the matched degree phrase. This means a field
  of study written unusually far from its degree mention, or a JD listing
  multiple degree levels in one sentence (e.g. *"Master's preferred,
  Bachelor's in Economics required"*), may attribute the field to the
  wrong (or no) degree level. Treat `education_fields` as a best-effort
  signal, not a guarantee.
- **Skill/role dictionaries are static lists.** Any skill or role phrasing
  not yet in `synonyms.py` won't be recognized — this is by design (no
  network calls, fully deterministic, and human-auditable), but it means
  the tables need periodic extension as new postings surface new
  terminology, the same tradeoff `resume_extraction_engine`'s section-alias
  table makes.
- **Experience parsing** targets the common "X years", "X+ years", and
  "X-Y years" phrasings. Unusual phrasing (e.g. "a decade of experience",
  "fresh graduates welcome") isn't parsed into a number and instead
  surfaces as a `warnings` entry so it can be reviewed manually.
- **Plain text / Markdown input only.** PDF and DOCX postings need to be
  text-extracted first (see the *Usage* note above pointing at Day 5's
  readers) — this module deliberately doesn't duplicate that file-format
  logic.
