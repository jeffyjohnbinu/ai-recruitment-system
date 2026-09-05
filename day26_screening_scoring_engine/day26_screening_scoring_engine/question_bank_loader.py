"""
question_bank_loader.py
-----------------------
Day 26 deliverable — Zecpath AI Job Portal

Loads the static HR screening question bank from
`hr_screening_question_bank/1_hr_screening_question_dataset.json` and
`2_question_category_mapping.json` into typed, addressable Python objects
the scoring engine can consume.

Design:
- Reads the JSON files by relative path (this package sits next to
  `hr_screening_question_bank/` at the repo root, so a single
  `Path(__file__).parent.parent / "hr_screening_question_bank"` works
  regardless of the caller's CWD).
- Alias-tolerant: each field can be referenced by either its canonical
  name or any reasonable variation, so a future schema bump won't
  silently zero out scores.
- All functions are pure / read-only: no writes, no side effects beyond
  caching the loaded bank on first import.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger("day26_screening_scoring_engine.question_bank_loader")


# Path to the question bank directory (sibling of this package).
_BANK_DIR = Path(__file__).resolve().parent.parent / "hr_screening_question_bank"


@dataclass
class Question:
    """One question from the HR screening question dataset."""

    question_id: str
    role_id: str
    category: str
    text: str
    expected_answer_type: str  # "text" | "number" | "boolean" | "enum" | "date" | "duration"
    mandatory: bool
    scoring_weight: int  # 1-5
    placeholders: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "role_id": self.role_id,
            "category": self.category,
            "text": self.text,
            "expected_answer_type": self.expected_answer_type,
            "mandatory": self.mandatory,
            "scoring_weight": self.scoring_weight,
            "placeholders": self.placeholders,
        }


@dataclass
class CategoryMeta:
    """Per-category metadata from the category mapping file."""

    name: str
    purpose: str
    flow_position: int
    question_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "flow_position": self.flow_position,
            "question_ids": self.question_ids,
        }


@dataclass
class QuestionBank:
    """In-memory view of the entire question bank."""

    schema_version: str
    roles: List[str]
    questions: List[Question]
    categories: Dict[str, CategoryMeta]
    answer_types: List[str]
    scoring_scale: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "roles": self.roles,
            "questions": [q.to_dict() for q in self.questions],
            "categories": {k: v.to_dict() for k, v in self.categories.items()},
            "answer_types": self.answer_types,
            "scoring_scale": self.scoring_scale,
        }


# Alias maps for forward-compat with future schema bumps.
_ID_ALIASES = ("question_id", "id", "qid")
_ROLE_ALIASES = ("role_id", "role", "job_role")
_CATEGORY_ALIASES = ("category", "cat", "section")
_TEXT_ALIASES = ("text", "question_text", "prompt", "question")
_ANSWER_TYPE_ALIASES = ("expected_answer_type", "answer_type", "type")
_MANDATORY_ALIASES = ("mandatory", "required", "is_mandatory")
_WEIGHT_ALIASES = ("scoring_weight", "weight", "importance", "score_weight")


def _pick(record: Dict[str, Any], aliases: tuple[str, ...], default: Any = None) -> Any:
    """Return the first non-None value across the alias list."""
    for key in aliases:
        if key in record and record[key] is not None:
            return record[key]
    return default


def _extract_placeholders(text: str) -> List[str]:
    """Pull all `{placeholder}` tokens out of a question's text template."""
    import re

    return re.findall(r"\{([a-z_][a-z0-9_]*)\}", text or "")


@lru_cache(maxsize=1)
def _load_questions_cached() -> List[Question]:
    """Load and cache the question dataset file."""
    return _read_questions_file(_BANK_DIR / "1_hr_screening_question_dataset.json")


def _read_questions_file(path: Path) -> List[Question]:
    raw = json.loads(path.read_text(encoding="utf-8"))

    # The dataset file is a dict with a "questions" key.
    if isinstance(raw, dict):
        questions_raw = raw.get("questions") or raw.get("items") or []
        roles = list(raw.get("roles") or [])
    elif isinstance(raw, list):
        questions_raw = raw
        roles = []
    else:
        questions_raw = []
        roles = []

    questions: List[Question] = []
    for q in questions_raw:
        if not isinstance(q, dict):
            continue
        qid = str(_pick(q, _ID_ALIASES, "") or "")
        if not qid:
            continue
        text = str(_pick(q, _TEXT_ALIASES, "") or "")
        questions.append(
            Question(
                question_id=qid,
                role_id=str(_pick(q, _ROLE_ALIASES, "") or ""),
                category=str(_pick(q, _CATEGORY_ALIASES, "") or ""),
                text=text,
                expected_answer_type=str(_pick(q, _ANSWER_TYPE_ALIASES, "text") or "text"),
                mandatory=bool(_pick(q, _MANDATORY_ALIASES, False)),
                scoring_weight=int(_pick(q, _WEIGHT_ALIASES, 1) or 1),
                placeholders=_extract_placeholders(text),
            )
        )
    logger.info("Loaded %d questions from %s", len(questions), path.name)
    if roles:
        logger.info("Question bank roles: %s", roles)
    return questions


@lru_cache(maxsize=1)
def _load_categories_cached() -> Dict[str, CategoryMeta]:
    """Load and cache the category mapping file."""
    return _read_categories_file(_BANK_DIR / "2_question_category_mapping.json")


def _read_categories_file(path: Path) -> Dict[str, CategoryMeta]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    categories_raw = raw.get("categories") or {}
    out: Dict[str, CategoryMeta] = {}
    for cat_id, cat in categories_raw.items():
        if not isinstance(cat, dict):
            continue
        out[str(cat_id)] = CategoryMeta(
            name=str(cat.get("name") or cat_id),
            purpose=str(cat.get("purpose") or ""),
            flow_position=int(cat.get("flow_position") or 0),
            question_ids=list(cat.get("question_ids") or []),
        )
    logger.info("Loaded %d category entries from %s", len(out), path.name)
    return out


@lru_cache(maxsize=1)
def load_question_bank() -> QuestionBank:
    """Return the full question bank, cached for the process lifetime."""
    questions = _load_questions_cached()
    categories = _load_categories_cached()
    # Derive roles from questions if not provided.
    roles = sorted({q.role_id for q in questions if q.role_id})
    scoring_scale: Dict[str, str] = {}
    answer_types: List[str] = []
    try:
        raw2 = json.loads(
            (_BANK_DIR / "2_question_category_mapping.json").read_text(encoding="utf-8")
        )
        scoring_scale = dict(raw2.get("scoring_scale") or {})
        answer_types = list(raw2.get("answer_types") or [])
    except Exception:  # noqa: BLE001
        pass

    return QuestionBank(
        schema_version="1.0",
        roles=roles,
        questions=questions,
        categories=categories,
        answer_types=answer_types,
        scoring_scale=scoring_scale,
    )


# Convenience module-level constant: the loaded bank.
QUESTION_BANK: QuestionBank = load_question_bank()


def find_question(question_id: str) -> Optional[Question]:
    """Return the Question with this id, or None if not in the bank."""
    for q in QUESTION_BANK.questions:
        if q.question_id == question_id:
            return q
    return None


def questions_for_role(role_id: str) -> List[Question]:
    """Return the questions in this role, ordered by their category's flow_position."""
    bank = QUESTION_BANK
    flow_position = {cid: meta.flow_position for cid, meta in bank.categories.items()}
    qs = [q for q in bank.questions if q.role_id == role_id]
    qs.sort(
        key=lambda q: (flow_position.get(q.category, 99), q.question_id),
    )
    return qs
