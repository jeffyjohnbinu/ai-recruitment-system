# Resume Section Classifier

**Day 8 deliverable — Zecpath AI Job Portal**

Segments cleaned resume text into labeled section blocks (Skills,
Experience, Education, Certifications, Projects, ...) so downstream
matching/screening modules can work on structured, meaningful content
instead of one undifferentiated blob of text.

Designed to sit directly downstream of **Day 5**'s
`resume_extraction_engine`: feed it `ExtractionRecord.cleaned_text`
and it tags every part of that text with a section label, using the
same canonical section names Day 5 already normalizes headings to.

## What it does

1. **Rule-based heading detection** — walks the cleaned text line by
   line, matching known section headings (and common variants:
   "WORK EXPERIENCE", "Technical Skills:", "Key Projects", ...)
   against a canonical alias table shared in spirit with Day 5's
   `_SECTION_ALIASES`. A matched heading opens a new block with
   100% confidence.
2. **NLP-style keyword fallback** — for text with no heading at all
   (some resumes never label sections), or the header/contact block
   at the very top, a lightweight weighted-keyword scorer classifies
   each paragraph independently (Skills vs. Experience vs. Education
   vs. Certifications vs. Projects vs. Summary vs. Achievements vs.
   Languages vs. Contact), boosted by structural signals like date
   ranges, degree abbreviations, and bullet density.
3. **Cross-validation** — even heading-matched blocks are re-scored
   by the NLP classifier in the background; if the heading and the
   content disagree (e.g. a "Skills" heading sitting over what reads
   entirely like work-experience bullets — a common copy/paste
   mistake in real resumes), the block is flagged in `warnings`
   instead of being silently trusted.
4. **Handles tables & columns** — because it consumes Day 5's
   already-linearized `cleaned_text` (multi-column PDFs reconstructed
   left-then-right, DOCX tables flattened to `Cell | Cell | Cell`
   rows), table/column layouts fall out "for free": a `Skills` table
   row stays inside the `Skills` block.
5. **Structured + human-readable storage** — same convention as
   Day 5/6: JSON for pipeline consumption, plain labeled text for
   manual review.

## Project layout

```
resume_section_classifier/
├── __init__.py
├── rules.py               # canonical labels + heading alias table + heading matcher
├── nlp_classifier.py        # weighted-keyword classifier for headingless content
├── segmenter.py               # two-pass segmentation (rule-based + NLP fallback)
├── classifier.py                # main orchestrator (SectionClassifierEngine)
├── storage.py                     # SectionBlockRecord/SectionClassificationRecord + ResultStore
├── cli.py                           # command-line entry point
└── tests/
    ├── test_section_classifier.py   # pytest suite (18 tests)
    ├── run_tests.py                   # runs the suite + writes a timestamped log
    ├── fixtures/                       # 5 labeled sample resumes + ground_truth.json
    ├── reports/                          # accuracy_report.md (generated, deliverable)
    └── logs/                               # generated test-run logs
```

## Requirements

Core module has **zero required third-party dependencies** — the
NLP fallback is a plain-Python keyword scorer, not a downloaded model,
so it runs offline with no network access.

```
pip install pytest --break-system-packages
```

Optional, only needed if you want `classify_file()` / the CLI to read
`.pdf`/`.docx` directly instead of going through Day 5's engine first:
```
pip install pdfplumber python-docx --break-system-packages
```

## Usage

### Python API — recommended integration with Day 5

```python
from resume_extraction_engine import ResumeExtractionEngine
from resume_section_classifier import SectionClassifierEngine

extractor = ResumeExtractionEngine(output_dir="extracted")
classifier = SectionClassifierEngine(output_dir="sectioned")

extraction = extractor.process_file("resumes/john_doe.pdf")
sections = classifier.classify_text(extraction.cleaned_text, source_name=extraction.source_file)

print(sections.sections_detected)   # ["Header", "Summary", "Experience", "Education", ...]
for block in sections.blocks:
    print(block.label, block.method, block.confidence, len(block.text))

# Ready-to-consume view for a downstream matcher:
payload = sections.as_matching_engine_payload()
print(payload["Skills"])
```

### Python API — standalone on a text/pdf/docx file

```python
from resume_section_classifier import SectionClassifierEngine

engine = SectionClassifierEngine(output_dir="outputs")
record = engine.process_file("resumes/john_doe.txt")
```

### CLI

```
python -m resume_section_classifier.cli resumes/john_doe.txt
python -m resume_section_classifier.cli resumes/ --output-dir sectioned/
```

### Output structure

For each resume, two files are written:
- `outputs/structured/<name>.sections.json` — full
  `SectionClassificationRecord` (every block: label, method,
  confidence, line range, text, cross-check flags)
- `outputs/labeled_text/<name>.labeled.txt` — plain text with each
  block prefixed by a `===== [Label] =====` header, for human review

## Running tests

```
python resume_section_classifier/tests/run_tests.py
```

Runs the full pytest suite (18 tests) and writes a timestamped log to
`tests/logs/test_run_<UTC timestamp>.log`. Also regenerates
`tests/reports/accuracy_report.md` — the section-detection accuracy
deliverable — scored against 5 hand-labeled fixture resumes covering:
full canonical headings, no headings at all, partial headings
(some sections labeled, some not), table-linearized skills content,
and noisy/OCR-style irregular spacing.

**Current accuracy: 28/29 labeled ground-truth snippets (97%)** across
the fixture set (see `tests/reports/accuracy_report.md` for the
per-fixture breakdown and the one documented mismatch).

## Known limitations

- **Keyword-based NLP fallback, not a trained classifier.** The
  scorer is transparent and tunable but will occasionally mislabel a
  short, low-signal paragraph (see the one documented mismatch in the
  accuracy report — a short bullet with no strong keyword or
  structural signal). A future iteration could add a small trained
  classifier (e.g. `sentence-transformers`, already in
  `requirements.txt`) as a second opinion when the keyword scorer's
  confidence is low.
- **Heading detection is alias-table based**, same tradeoff as Day
  5's heading normalization: very unusual heading phrasing not in the
  alias table falls through to the NLP fallback rather than being
  recognized as a heading.
- **`classify_file()` on raw `.pdf`/`.docx`** uses a simple text
  extraction path (no multi-column reconstruction, no OCR fallback).
  For real-world PDFs/DOCX files, route them through Day 5's
  `resume_extraction_engine` first and call `classify_text()` on its
  `cleaned_text` output, as shown above.
