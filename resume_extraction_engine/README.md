# Resume Text Extraction Engine

**Day 5 deliverable — Zecpath AI Job Portal**

Converts raw resume files (PDF / DOCX) into clean, normalized, structured
text that downstream AI modules (parsing, scoring, matching) can consume
reliably.

## What it does

1. **Reads** PDF and DOCX resumes, including tables, multi-column layouts,
   and flags embedded images.
2. **Cleans** the extracted text: strips control characters and stray
   symbols, collapses whitespace/blank-line noise, de-hyphenates
   line-wrapped words, drops page-number artifacts.
3. **Normalizes** capitalization (fixes "shouted" all-caps sentences while
   preserving real acronyms like AWS, SQL, PMP), unifies bullet styles into
   a single `- ` format, and canonicalizes section headings (e.g. "WORK
   EXPERIENCE" / "Employment History" → `Experience`).
4. **Stores** results as structured JSON (for pipeline consumption) and
   plain cleaned `.txt` (for human review).
5. Ships with an **automated test suite** and a runner that writes
   timestamped log files.

## Project layout

```
resume_extraction_engine/
├── __init__.py
├── extractor.py          # main orchestrator (ResumeExtractionEngine)
├── cleaner.py             # TextCleaner + TextNormalizer
├── storage.py              # ExtractionRecord + ResultStore (JSON/txt output)
├── cli.py                    # command-line entry point
├── readers/
│   ├── pdf_reader.py       # pdfplumber (primary) + PyPDF2 (fallback), OCR-ready
│   └── docx_reader.py      # python-docx, preserves paragraph/table order
└── tests/
    ├── test_extraction.py  # pytest suite (16 tests)
    ├── run_tests.py         # runs the suite + writes a timestamped log
    ├── sample_resumes/       # 1 DOCX + 2 PDFs used as test fixtures
    └── logs/                  # generated test-run logs (deliverable)
```

## Requirements

```
pip install pdfplumber PyPDF2 python-docx pytest --break-system-packages
```

Optional, for OCR fallback on image-only PDF pages:
```
pip install pytesseract pillow --break-system-packages
```
(also requires the `tesseract-ocr` system binary; if unavailable, the
engine degrades gracefully and simply flags the page as containing
un-extracted image content instead of crashing.)

## Usage

### Python API

```python
from resume_extraction_engine import ResumeExtractionEngine

engine = ResumeExtractionEngine(output_dir="outputs")
record = engine.process_file("resumes/john_doe.pdf")

print(record.status)            # "success" | "partial" | "failed"
print(record.sections_detected) # ["Summary", "Experience", "Education", ...]
print(record.cleaned_text)      # normalized, ready-for-AI text
```

Batch processing a folder:
```python
records = engine.process_directory("resumes/")
```

### CLI

```
python -m resume_extraction_engine.cli resumes/john_doe.pdf
python -m resume_extraction_engine.cli resumes/ --output-dir extracted/
```

### Output structure

For each resume, two files are written:
- `outputs/structured/<name>.json` — full `ExtractionRecord` (metadata,
  warnings, cleaning stats, detected sections, cleaned text)
- `outputs/cleaned_text/<name>.clean.txt` — plain cleaned text only

## Running tests

```
python resume_extraction_engine/tests/run_tests.py
```

This runs the full pytest suite and writes a timestamped log to
`tests/logs/test_run_<UTC timestamp>.log`, in addition to printing the
result to stdout. 16 tests cover: DOCX text/table/bullet/section
extraction, PDF two-column layout reconstruction (with a regression guard
against column interleaving), noisy-PDF cleanup (de-hyphenation, page-number
removal, glyph-substitution bullet handling), error handling for missing or
unsupported files, and isolated cleaner/normalizer unit behavior.

## Known limitations

- **Column detection** uses a positional heuristic (word x-coordinates vs.
  page midpoint). It handles standard 2-column resume templates well but
  is not a general-purpose layout engine — highly irregular or 3+ column
  layouts may need manual review.
- **Image-only PDFs** (scanned resumes) require the optional OCR path
  (`pytesseract` + `tesseract-ocr`); without it, image pages are flagged
  in `warnings` rather than transcribed.
- **Font glyph substitution**: some PDF exporters embed custom/subset
  fonts without a proper Unicode mapping for symbol characters (e.g. a
  bullet glyph rendering as literal `(cid:127)`). The cleaner recognizes
  the common `(cid:N)` artifact and normalizes it to a bullet, but exotic
  fonts may still produce unrecognized characters.
