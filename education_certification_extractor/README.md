# Education & Certification Parsing Engine

**Day 11 deliverable — Zecpath AI Job Portal**

Detects academic qualifications and professional certifications from
resume text and turns them into a structured academic profile that
downstream modules (ATS matching, screening, interview prep) can consume
without re-parsing free text.

## What it does

1. **Locates** the Education and Certifications content, either from a
   pre-segmented section dict (e.g. Day 8's `resume_section_classifier`
   output) or by scanning a flat cleaned-text block (e.g. Day 5's
   `ExtractionRecord.cleaned_text`) for the same canonical section
   headings the rest of the pipeline already recognizes.
2. **Extracts degrees**: degree type, field of study, institution, and
   graduation year, using a two-pass strategy — a structured regex pass
   for the common "Degree in Field — Institution, Year" template, and a
   looser fallback pass for irregular formatting (e.g. degree on one
   line, institution/year on the next).
3. **Extracts certifications**: name, issuer (when stated), and year
   (when stated).
4. **Normalizes naming**: collapses abbreviation variants ("B.S.", "BS",
   "Bachelors of Science") into one canonical degree name, and tidies
   institution/certification text.
5. **Tags certification relevance**: assigns each certification a
   category (Cloud, Security, Data & AI, Project Management, Agile &
   Scrum, Networking & Systems, Quality & Process, Finance & Business,
   Programming & Development, or Other) via keyword lookup.
6. **Computes `highest_degree`** across all parsed degrees using a
   seniority ranking (PhD > Master's > Bachelor's > Associate > Diploma).
7. **Stores** results as structured JSON, one file per resume, following
   the same `outputs/structured/<name>.json` convention as Day 5.

## Project layout

```
education_certification_extractor/
├── __init__.py
├── extractor.py              # main orchestrator (EducationCertificationExtractor)
├── degree_parser.py           # DegreeParser: two-pass degree/field/institution/year extraction
├── certification_parser.py     # CertificationParser: name/issuer/year extraction
├── normalizer.py                # degree/institution/certification-name canonicalization
├── relevance_tagger.py           # keyword-based certification category tagging
├── storage.py                     # AcademicProfileRecord + ResultStore (JSON output)
├── cli.py                          # command-line entry point
└── tests/
    ├── test_extraction.py          # pytest suite (23 tests)
    ├── run_tests.py                 # runs the suite + writes a timestamped log
    ├── sample_data/                  # 3 fixture resumes (clean, multi-degree, noisy)
    └── logs/                          # generated test-run logs (deliverable)
```

## Requirements

Standard library only for the extraction logic; `pytest` for the test
suite:

```
pip install pytest --break-system-packages
```

## Usage

### Python API — from pre-segmented sections (Day 8 output)

```python
from education_certification_extractor import EducationCertificationExtractor

engine = EducationCertificationExtractor(output_dir="outputs")
record = engine.extract_from_sections(
    {
        "Education": "B.S. in Computer Science — University of Texas at Austin, 2018",
        "Certifications": "- AWS Certified Solutions Architect – Associate",
    },
    source_file="john_doe.docx",
)

print(record.status)            # "success" | "partial" | "failed"
print(record.highest_degree)    # "Bachelor of Science"
print(record.degrees)           # [DegreeRecord(...)]
print(record.certifications)    # [CertificationRecord(relevance_category="Cloud", ...)]
```

### Python API — from flat cleaned text (Day 5 output)

```python
record = engine.extract_from_text(cleaned_text, source_file="john_doe.pdf")
```

### CLI

```
python -m education_certification_extractor.cli resume.clean.txt
python -m education_certification_extractor.cli cleaned_text/ --output-dir extracted/
```

### Output structure

For each resume, one file is written:
- `outputs/structured/<name>.academic_profile.json` — full
  `AcademicProfileRecord` (degrees, certifications, highest degree,
  warnings, status)

## Running tests

```
python education_certification_extractor/tests/run_tests.py
```

This runs the full pytest suite and writes a timestamped log to
`tests/logs/test_run_<UTC timestamp>.log`. 23 tests cover: degree-name
normalization, certification relevance tagging, the structured and
fallback degree-parsing passes (including a degree split across two
lines), certification name/issuer/year extraction, alias-heading
matching, and end-to-end extraction from three fixture resumes (clean
single-degree, multi-degree with four certifications, and a noisy/
irregular education section).

## Known limitations & design notes

- **Section-boundary heuristic in `extract_from_text`**: when only flat
  cleaned text is available (no Day 8 segmentation), a line is treated
  as a section heading only if it *exactly* matches a known heading and
  is under 40 characters — matching the same heuristic Day 5's cleaner
  uses for heading normalization, for consistency. Resumes with truly
  unlabeled section breaks will need Day 8's classifier for reliable
  segmentation; `extract_from_sections` is the more robust entry point
  and is the one downstream modules should prefer once Day 8 output is
  available in the pipeline.
- **Skills sub-category noise (bug caught during testing)**: some
  cleaned resumes emit unheaded skills sub-lines like
  `Languages | Python, Java, SQL` directly after the Certifications
  heading (no dedicated "Skills" content precedes them). Early testing
  showed these getting swept into the Certifications buffer and
  misparsed as bogus certification entries. Fixed by having
  `CertificationParser` skip any line containing `" | "`, since that
  pipe-delimited format is specific to the skills-breakdown convention
  used elsewhere in this pipeline and never appears in a legitimate
  certification line.
- **Degree seniority ranking** is a fixed list (`_DEGREE_RANK` in
  `extractor.py`) rather than a scored heuristic. It covers the degree
  types produced by `normalizer.normalize_degree()`; an unrecognized
  degree type is excluded from `highest_degree` consideration rather
  than guessed at.
- **Confidence scores** are heuristic buckets (0.95 / 0.6 / 0.3 for
  degrees; 0.85 / 0.7 for certifications) reflecting which parsing pass
  matched, not a calibrated probability — consistent with how Day 8
  handled NLP-fallback confidence.
