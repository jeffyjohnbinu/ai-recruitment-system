# AI Recruitment System

An AI-assisted recruitment pipeline: parses resumes, matches them against job
descriptions (ATS engine), runs AI-driven candidate screening, generates
interview questions, and produces a final ranked candidate score.

This repository is the **Day 3 deliverable**: environment setup, folder
structure, logging, sample tests, and code standards. Each module folder
contains a working scaffold so subsequent days can build on real code rather
than empty directories.

---

## 1. Project Layout

```
ai-recruitment-system/
├── data/                 # Resume/job-description data
│   ├── raw/               # Original input files (gitignored, sample files kept)
│   └── processed/         # Cleaned/derived data (gitignored)
├── parsers/               # Resume/document parsing (PDF, DOCX -> structured text)
│   └── resume_parser.py
├── ats_engine/             # Applicant Tracking System: resume <-> job matching
│   └── matcher.py
├── screening_ai/           # LLM-based candidate screening & summarization
│   └── screener.py
├── interview_ai/            # Interview question generation & response evaluation
│   └── question_generator.py
├── scoring/                # Combines ATS + AI screening into a final rank
│   └── scorer.py
├── utils/                  # Shared helpers: logging, validators
│   ├── logger.py
│   └── validators.py
├── config/                 # Centralized settings, loaded from .env
│   └── settings.py
├── tests/                  # Pytest test suite (mirrors module structure)
├── logs/                   # Runtime logs (gitignored; created automatically)
├── docs/
│   └── CODE_STANDARDS.md   # Coding conventions & documentation format
├── main.py                 # Demo pipeline entry point
├── requirements.txt        # Runtime dependencies
├── requirements-dev.txt    # Dev-only tooling
├── setup_env.sh / .ps1     # One-command environment bootstrap
├── .env.example             # Environment variable template
├── pytest.ini / pyproject.toml / .flake8   # Test & lint configuration
└── .pre-commit-config.yaml
```

**Design principle:** each numbered folder is a pipeline stage. Data flows
left to right: `parsers` → `ats_engine` → `screening_ai` → `interview_ai` →
`scoring`, with `utils` and `config` shared across all of them. This keeps
each stage independently testable and replaceable (e.g. swap the keyword
matcher in `ats_engine` for a semantic embedding matcher without touching
anything else).

---

## 2. Environment Setup

### Requirements
- Python 3.11+ (check with `python3 --version`)
- Git

### Quick start (macOS/Linux)
```bash
git clone <your-repo-url>
cd ai-recruitment-system
bash setup_env.sh          # creates .venv, installs requirements.txt
# or, including dev tools + pre-commit hooks:
bash setup_env.sh --dev
```

### Quick start (Windows / PowerShell)
```powershell
git clone <your-repo-url>
cd ai-recruitment-system
powershell -ExecutionPolicy Bypass -File setup_env.ps1
```

### Manual setup (any OS)
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env             # then fill in OPENAI_API_KEY etc.
```

---

## 3. Running the Demo Pipeline

A sample resume and job description are included under `data/raw/`:

```bash
python main.py --resume data/raw/sample_resume.txt --job data/raw/sample_job.txt
```

Expected output:
```
--- Pipeline Result ---
ATS match score : 0.7x
Matched keywords: aws, docker, fastapi, postgresql, pytest, python
AI recommendation: advance
Final score      : 0.8x
```

The demo skips the live LLM call in `screening_ai/screener.py` by default
(no API key required to try the pipeline). Once `OPENAI_API_KEY` is set in
`.env`, wire `screen_candidate()` into `main.py` for full AI screening.

---

## 4. Logging

All modules log through `utils/logger.py`, configured centrally from
`config/settings.py` (itself driven by `.env`):

- Console output for local development.
- `logs/app.log` — rotating daily log, 14-day retention.
- `logs/errors.log` — errors/criticals only, for fast triage.

```python
from utils.logger import get_logger
logger = get_logger(__name__)
logger.info("Parsed resume for candidate %s", candidate_id)
```

Control verbosity with `LOG_LEVEL` in `.env` (`DEBUG`, `INFO`, `WARNING`,
`ERROR`, `CRITICAL`).

---

## 5. Testing

```bash
pytest                          # run full suite
pytest --cov=. --cov-report=term-missing   # with coverage
pytest tests/test_matcher.py -v            # single file
```

The suite in `tests/` mirrors the module structure (`test_matcher.py`,
`test_scorer.py`, `test_validators.py`, `test_logger.py`) and uses shared
fixtures from `tests/conftest.py`. Network-dependent code (LLM calls) is
isolated behind small functions so it can be mocked — see
`docs/CODE_STANDARDS.md` §5.

---

## 6. Code Standards

Full conventions (naming, docstring format, logging rules, commit style,
branching model) live in [`docs/CODE_STANDARDS.md`](docs/CODE_STANDARDS.md).
Summary:

```bash
black . && isort . && flake8 && mypy .    # format, sort imports, lint, type-check
pre-commit install                          # or let git hooks do it automatically
```

---

## 7. Next Steps (beyond Day 3)

- Replace the keyword-overlap matcher in `ats_engine/matcher.py` with
  semantic matching via `sentence-transformers`.
- Add structured (JSON-mode) output to `screening_ai/screener.py`.
- Build a FastAPI layer (`fastapi` is already in `requirements.txt`) to
  expose the pipeline as an API.
- Add a `data/processed/` persistence layer (SQLite/Postgres via
  `DATABASE_URL` in `.env`).
