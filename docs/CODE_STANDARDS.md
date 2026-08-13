# Code Standards & Documentation Format

This document defines the conventions every contributor (human or AI-assisted)
follows in this repository.

## 1. Style & Formatting

- **Formatter:** [`black`](https://black.readthedocs.io/), line length 100.
- **Import order:** [`isort`](https://pycqa.github.io/isort/) with the `black` profile.
- **Linting:** `flake8` must pass with zero warnings before merge.
- **Type checking:** `mypy` is run in CI; add type hints to all new public functions.
- Run all four locally with:
  ```bash
  black . && isort . && flake8 && mypy .
  ```
- Or let `pre-commit` do it automatically on every commit (`pre-commit install`).

## 2. Naming Conventions

| Element         | Convention             | Example                     |
|------------------|------------------------|------------------------------|
| Modules/files    | `snake_case.py`        | `resume_parser.py`          |
| Classes          | `PascalCase`           | `ParsedResume`               |
| Functions/vars   | `snake_case`           | `match_resume_to_job()`     |
| Constants        | `UPPER_SNAKE_CASE`     | `ATS_MIN_MATCH_SCORE`       |
| Private helpers  | leading underscore     | `_find_phone_candidate()`   |

## 3. Docstring Format

Every module, class, and public function gets a docstring. Modules use a
header block; functions use Google-style docstrings:

```python
def match_resume_to_job(resume_text: str, job_description: str) -> MatchResult:
    """
    Compute a keyword-overlap match score between a resume and job description.

    Args:
        resume_text: Raw text extracted from the candidate's resume.
        job_description: Raw text of the job posting.

    Returns:
        A MatchResult containing the score, matched keywords, and shortlist flag.
    """
```

## 4. Logging Rules

- Never use `print()` in library code (`parsers/`, `ats_engine/`, `screening_ai/`,
  `interview_ai/`, `scoring/`, `utils/`). Use `utils.logger.get_logger(__name__)`.
- `print()` is acceptable only in `main.py` / CLI-facing output.
- Log levels:
  - `DEBUG` — verbose internal state, disabled by default in production.
  - `INFO` — normal pipeline milestones (parsed resume, computed score).
  - `WARNING` — recoverable issues (missing field, low-confidence match).
  - `ERROR` — operation failed but the process continues (use `exc_info=True`).
  - `CRITICAL` — the process cannot continue safely.

## 5. Testing Rules

- Every new module in `parsers/`, `ats_engine/`, `screening_ai/`, `interview_ai/`,
  or `scoring/` gets a matching `tests/test_<module>.py`.
- Network calls (OpenAI, external APIs) must be isolated behind a small function
  (see `screening_ai/screener.py::_call_llm`) so tests can mock them with
  `pytest-mock` instead of hitting the network.
- Target meaningful coverage of business logic, not 100% line coverage for its
  own sake. Run coverage with:
  ```bash
  pytest --cov=. --cov-report=term-missing
  ```

## 6. Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(ats_engine): add semantic matching with sentence-transformers
fix(parsers): handle empty PDF pages without crashing
docs(readme): update setup instructions
test(scoring): add edge cases for tied final scores
```

## 7. Branching Model

- `main` — always deployable.
- `feature/<short-description>` — new work, opened as a PR into `main`.
- `fix/<short-description>` — bug fixes.
- PRs require: passing tests, passing lint/type-check, and one review.

## 8. Configuration & Secrets

- All configuration lives in `config/settings.py`, sourced from environment
  variables (`.env`, never committed — see `.env.example`).
- Never hardcode API keys, credentials, or file paths inside modules.
