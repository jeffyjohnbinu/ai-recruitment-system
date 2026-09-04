"""
answer_understanding_engine/structured_answer_format.py
------------------------------------------------------
Day 25 — Structured Answer Format (standalone module).

A dedicated, focused module that defines the structured output format
produced by the Answer Understanding Engine.

Provides the canonical `StructuredAnswer` dataclass along with all
supporting types (`Slot`, `IntentResult`) and the helper methods needed
to consume the output.

Public API:
    from answer_understanding_engine.structured_answer_format import (
        StructuredAnswer,
        Slot,
        IntentResult,
        AnswerQuality,
    )

Usage:
    result = engine.understand(answer_text, question_id="Q-SE-006")
    print(result.to_dict())         # JSON-serializable dict
    print(result.is_off_topic)     # bool
    print(result.extracted)        # {slot_name: value}
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

# --------------------------------------------------------------------------- #
# Supporting types                                                             #
# --------------------------------------------------------------------------- #


@dataclass
class IntentResult:
    """
    Classification of what the candidate's reply is *doing*.

    Attributes
    ----------
    label : str
        One of: "answer", "clarification_request", "objection",
        "redirect", "off_topic", "no_response".
    confidence : float
        Confidence score in the range [0.0, 1.0].
    rationale : str
        A short, human-readable explanation of why this label was chosen.
    """

    label: str
    confidence: float
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Slot:
    """
    A single piece of information extracted from the candidate's reply.

    Attributes
    ----------
    name : str
        Canonical slot name. Common values:
        - ``boolean``             yes/no answer (``True`` / ``False``)
        - ``years_experience``    years of professional experience (float)
        - ``notice_period_days``  notice period in days (int)
        - ``availability``         "immediate" or other text
        - ``join_by_date``        date string as written (e.g. "15/03/2026")
        - ``expected_salary_lakhs`` annual salary in INR lakhs (float)
        - ``skills``              list of recognized skill names
        - ``city``                canonicalized city name
    value : Any
        The extracted value (type depends on ``name``).
    raw_span : str
        The exact substring from the original answer that was used to
        produce this value. Useful for explainability / auditability.
    confidence : float
        Confidence score in the range [0.0, 1.0].
    """

    name: str
    value: Any
    raw_span: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnswerQuality:
    """
    Per-answer quality indicators produced by the quality assessor.

    Attributes
    ----------
    is_off_topic : bool
        True if the reply is clearly unrelated to the question.
    is_vague : bool
        True if the reply contains vague qualifiers
        (e.g. "maybe", "kind of", "around 3 years").
    is_missing : bool
        True if no useful answer was given
        (empty, filler, or the expected slot was not extracted).
    completeness : float
        A 0.0–1.0 score where 1.0 means the answer fully satisfies the
        question's intent and provides the expected slot.
    warnings : List[str]
        Human-readable warning messages (e.g. "Low intent confidence",
        "Expected slot 'years_experience' was not extracted").
    """

    is_off_topic: bool = False
    is_vague: bool = False
    is_missing: bool = False
    completeness: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Primary output type                                                         #
# --------------------------------------------------------------------------- #


@dataclass
class StructuredAnswer:
    """
    Final structured output of the Answer Intent & Understanding Engine.

    Produced by ``AnswerUnderstandingEngine.understand()`` for every
    candidate reply. Fully JSON-serializable via ``to_dict()``.

    Attributes
    ----------
    question_id : str
        The ID of the screening question that was answered
        (e.g. "Q-SE-006").
    raw_answer : str
        The original, unprocessed answer text.
    intent : IntentResult
        The intent classification of the reply.
    slots : List[Slot]
        All extracted information pieces.
    extracted : Dict[str, Any]
        Convenience accessor: ``{slot_name: slot_value}`` for direct
        lookups (e.g. ``result.extracted["years_experience"]``).
    is_off_topic : bool
        Convenience alias for ``quality.is_off_topic``.
    is_vague : bool
        Convenience alias for ``quality.is_vague``.
    is_missing : bool
        Convenience alias for ``quality.is_missing``.
    completeness : float
        Convenience alias for ``quality.completeness``.
    warnings : List[str]
        Convenience alias for ``quality.warnings``.

    Examples
    --------
    >>> result = engine.understand(
    ...     "I have 5 years of experience in Python and AWS.",
    ...     question_id="Q-SE-006",
    ...     expected_slot="years_experience",
    ... )
    >>> result.intent.label
    'answer'
    >>> result.extracted["years_experience"]
    5.0
    >>> result.extracted["skills"]
    ['python', 'aws']
    >>> result.completeness
    1.0
    >>> import json; print(json.dumps(result.to_dict(), indent=2))
    {
      "question_id": "Q-SE-006",
      "raw_answer": "I have 5 years...",
      "intent": {"label": "answer", "confidence": 0.8, "rationale": "..."},
      "slots": [...],
      "extracted": {"years_experience": 5.0, "skills": ["python", "aws"]},
      "is_off_topic": false,
      "is_vague": false,
      "is_missing": false,
      "completeness": 1.0,
      "warnings": []
    }
    """

    question_id: str
    raw_answer: str
    intent: IntentResult
    slots: List[Slot]
    extracted: Dict[str, Any]
    is_off_topic: bool = False
    is_vague: bool = False
    is_missing: bool = False
    completeness: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable dict representation.

        All nested objects (``IntentResult``, ``Slot``) are also
        converted via their own ``to_dict()`` methods.
        """
        return {
            "question_id": self.question_id,
            "raw_answer": self.raw_answer,
            "intent": self.intent.to_dict(),
            "slots": [s.to_dict() for s in self.slots],
            "extracted": self.extracted,
            "is_off_topic": self.is_off_topic,
            "is_vague": self.is_vague,
            "is_missing": self.is_missing,
            "completeness": round(self.completeness, 2),
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StructuredAnswer:
        """
        Reconstruct a ``StructuredAnswer`` from a dict (e.g. loaded from JSON).

        Inverse of ``to_dict()``.
        """
        return cls(
            question_id=data["question_id"],
            raw_answer=data["raw_answer"],
            intent=IntentResult(**data["intent"]),
            slots=[Slot(**s) for s in data["slots"]],
            extracted=data["extracted"],
            is_off_topic=data.get("is_off_topic", False),
            is_vague=data.get("is_vague", False),
            is_missing=data.get("is_missing", False),
            completeness=data.get("completeness", 0.0),
            warnings=data.get("warnings", []),
        )


__all__ = [
    "IntentResult",
    "Slot",
    "AnswerQuality",
    "StructuredAnswer",
]
