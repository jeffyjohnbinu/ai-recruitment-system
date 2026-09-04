"""
answer_understanding_engine/engine.py
---------------------------------------
Day 25 deliverable — Answer Intent & Understanding Engine.

Turns a candidate's free-form reply to a screening question into a
structured semantic object: intent label, slot values, confidence, and
quality flags (off-topic / vague / missing).

Design:
- Pattern-first, LLM-optional. Default behavior is deterministic regex
  + vocabulary matching, so the module is testable without an API key.
- A `_call_llm` hook is provided for upgrades (swap in a structured
  JSON-mode prompt), mirroring screening_ai.screener._call_llm.
- Output is a `StructuredAnswer` dataclass with slots for the four
  question categories the screening bank cares about: skills,
  experience, availability, salary.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger("answer_understanding_engine.engine")


# --- public dataclasses ---------------------------------------------------- #


@dataclass
class IntentResult:
    """Classification of what the candidate's reply is *doing*."""

    label: str
    confidence: float
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Slot:
    """A single piece of information extracted from the answer."""

    name: str
    value: Any
    raw_span: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StructuredAnswer:
    """Final answer-understanding output for one candidate reply."""

    question_id: str
    raw_answer: str
    intent: IntentResult
    slots: List[Slot]
    extracted: Dict[str, Any]
    is_off_topic: bool
    is_vague: bool
    is_missing: bool
    completeness: float
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
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


# --- intent labels --------------------------------------------------------- #

_INTENT_LABELS = (
    "answer",
    "clarification_request",
    "objection",
    "redirect",
    "off_topic",
    "no_response",
)

_NO_RESPONSE_PATTERNS = [
    r"^\s*$",
    r"^(idk|i don'?t know|no idea|not sure|na|n/?a|none|nothing)\b",
    r"^(uh+|um+|hmm+|er+|ah+)\s*[.,!]?\s*$",
]

_OFF_TOPIC_RE = re.compile(
    r"\b("
    r"recipe|cricket score|movie review|weather forecast|"
    r"bitcoin price|stock tip|joke|poem|love letter|"
    r"buy my course|promote my (channel|product)|"
    r"(youtube|you.?tube) channel"
    r")\b",
    re.IGNORECASE,
)

_QUESTION_MARKERS = re.compile(
    r"\?$|^(what|why|how|when|where|who|can you|could you)\b",
    re.IGNORECASE,
)

_REDIRECT_PATTERNS = re.compile(
    r"\b(instead|rather (talk|discuss)|can we discuss|"
    r"let'?s talk about|skip|change the subject|move on|"
    r"next question)\b",
    re.IGNORECASE,
)

_OBJECTION_MARKERS = re.compile(
    r"\b(i (don'?t|do not) (want|like|feel)|"
    r"this (doesn'?t|does not) (feel|seem) right|"
    r"i'?m (not interested|uncomfortable|busy)|"
    r"why (should i|do i)|"
    r"this is (a waste|unfair|biased))\b",
    re.IGNORECASE,
)


# --- slot vocabulary ------------------------------------------------------- #

_YEARS_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(\+)?\s*(years?|yrs?|y\.?)\b",
    re.IGNORECASE,
)

_FRESHER_RE = re.compile(
    r"\b(fresher|fresh graduate|just graduated|0\s*years?)\b",
    re.IGNORECASE,
)

_DURATION_RE = re.compile(
    r"\b(\d+)\s*(day|week|month|months|days|weeks)s?\b",
    re.IGNORECASE,
)

_IMMEDIATE_RE = re.compile(
    r"\b(immediate(ly)?|right away|can join (today|now|asap)|"
    r"already serving notice|no notice)\b",
    re.IGNORECASE,
)

_DATE_RE = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"[a-z]*\s+\d{2,4}|"
    r"(next|this)\s+(week|month|monday))\b",
    re.IGNORECASE,
)

_MONEY_LPA_RE = re.compile(
    r"(?:rs\.?|inr|₹|\$)?\s*(\d+(?:\.\d+)?)\s*"
    r"(lpa|lakh|lakhs|l\b|lac|lacs|lakhs?\s*per\s*annum)\b",
    re.IGNORECASE,
)

_MONEY_K_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*k\b",
    re.IGNORECASE,
)

_MONEY_FULL_RE = re.compile(
    r"(?:rs\.?|inr|₹|\$)?\s*(\d[\d,]+\d|\d+)\s*" r"(per\s*annum|p\.?a\.?|annually|a\s*year)?\b",
    re.IGNORECASE,
)

_AFFIRMATIVE = re.compile(
    r"^(yes|yeah|yep|sure|of course|definitely|absolutely|"
    r"i (can|do|am|will)|happy to|no problem|i'?d love|"
    r"sounds good|that works|ok(ay)?)[.!]?\s*",
    re.IGNORECASE,
)

_NEGATIVE = re.compile(
    r"^(no|nope|nah|not really|i (can'?t|don'?t|won'?t|am not)|"
    r"unfortunately|afraid not|never)[.!]?\s*",
    re.IGNORECASE,
)


# Skills vocabulary
_SKILL_VOCAB = {
    "python",
    "java",
    "javascript",
    "typescript",
    "c++",
    "c#",
    "go",
    "golang",
    "rust",
    "ruby",
    "php",
    "kotlin",
    "swift",
    "scala",
    "r",
    "sql",
    "react",
    "angular",
    "vue",
    "next.js",
    "node.js",
    "html",
    "css",
    "redux",
    "fastapi",
    "django",
    "flask",
    "spring",
    "spring boot",
    "express",
    "graphql",
    "rest",
    "grpc",
    "microservices",
    "postgresql",
    "mysql",
    "mongodb",
    "redis",
    "elasticsearch",
    "kafka",
    "spark",
    "hadoop",
    "airflow",
    "snowflake",
    "bigquery",
    "aws",
    "gcp",
    "azure",
    "docker",
    "kubernetes",
    "terraform",
    "ansible",
    "jenkins",
    "github actions",
    "ci/cd",
    "helm",
    "tensorflow",
    "pytorch",
    "scikit-learn",
    "pandas",
    "numpy",
    "nlp",
    "machine learning",
    "deep learning",
    "computer vision",
    "pytest",
    "junit",
    "selenium",
    "cypress",
    "playwright",
    "salesforce",
    "hubspot",
    "zendesk",
    "freshdesk",
    "google analytics",
    "seo",
    "sem",
    "meta ads",
    "google ads",
    "negotiation",
    "communication",
    "excel",
    "tableau",
    "power bi",
    "looker",
}


# City aliases
_CITY_ALIASES = {
    "mumbai": "mumbai",
    "delhi": "delhi",
    "new delhi": "delhi",
    "bengaluru": "bengaluru",
    "bangalore": "bengaluru",
    "blr": "bengaluru",
    "hyderabad": "hyderabad",
    "hyd": "hyderabad",
    "chennai": "chennai",
    "madras": "chennai",
    "pune": "pune",
    "kolkata": "kolkata",
    "calcutta": "kolkata",
    "gurgaon": "gurgaon",
    "gurugram": "gurgaon",
    "noida": "noida",
    "ahmedabad": "ahmedabad",
    "jaipur": "jaipur",
    "lucknow": "lucknow",
    "kochi": "kochi",
    "coimbatore": "coimbatore",
    "indore": "indore",
    "remote": "remote",
}

_CITIES = list(_CITY_ALIASES.keys())

_VAGUE_MARKERS = re.compile(
    r"\b(maybe|kind of|sort of|approximately|around|"
    r"i think|i guess|it depends|roughly|approximately)\b",
    re.IGNORECASE,
)


# --- engine ---------------------------------------------------------------- #


class AnswerUnderstandingEngine:
    """
    Default rule-based answer understanding.

    Returns a StructuredAnswer with intent, slot values,
    and quality flags.
    """

    def __init__(self, use_llm: bool = False) -> None:
        self.use_llm = use_llm

    def understand(
        self,
        answer: str,
        question_id: str = "",
        expected_slot: Optional[str] = None,
    ) -> StructuredAnswer:
        """
        Run the full understanding pipeline on one answer.

        `expected_slot` is the slot name the question is trying to
        collect, for example years_experience, expected_salary,
        notice_period, or skills.
        """
        text = (answer or "").strip()

        logger.info(
            "Understanding answer for question_id=%s (len=%d)",
            question_id,
            len(text),
        )

        intent = self._classify_intent(text)
        slots = self._extract_slots(text, expected_slot=expected_slot)
        extracted = {s.name: s.value for s in slots}

        flags = self._assess_quality(
            text=text,
            intent=intent,
            slots=slots,
            expected_slot=expected_slot,
        )

        result = StructuredAnswer(
            question_id=question_id,
            raw_answer=text,
            intent=intent,
            slots=slots,
            extracted=extracted,
            is_off_topic=flags["is_off_topic"],
            is_vague=flags["is_vague"],
            is_missing=flags["is_missing"],
            completeness=flags["completeness"],
            warnings=flags["warnings"],
        )

        logger.info(
            "intent=%s slots=%s completeness=%.2f",
            intent.label,
            list(extracted.keys()),
            result.completeness,
        )

        return result

    def _classify_intent(self, text: str) -> IntentResult:
        """Pick the most likely intent label for the reply."""
        if not text:
            return IntentResult("no_response", 0.99, "Empty reply.")

        for pat in _NO_RESPONSE_PATTERNS:
            if re.match(pat, text, re.IGNORECASE):
                return IntentResult(
                    "no_response",
                    0.95,
                    "Filler or explicit no-info reply.",
                )

        if _OFF_TOPIC_RE.search(text):
            return IntentResult(
                "off_topic",
                0.9,
                "Reply contains an off-topic keyword.",
            )

        if _REDIRECT_PATTERNS.search(text):
            return IntentResult(
                "redirect",
                0.75,
                "Candidate tried to change the subject.",
            )

        if _OBJECTION_MARKERS.search(text):
            return IntentResult(
                "objection",
                0.75,
                "Candidate raised a concern / objection.",
            )

        if _QUESTION_MARKERS.search(text):
            return IntentResult(
                "clarification_request",
                0.7,
                "Reply is itself a question.",
            )

        return IntentResult(
            "answer",
            0.8,
            "Treating reply as an on-topic answer.",
        )

    def _extract_slots(
        self,
        text: str,
        expected_slot: Optional[str],
    ) -> List[Slot]:
        """
        Pull every recognized slot out of the reply.

        The optional LLM path is added on top of these patterns.
        """
        slots: List[Slot] = []

        # Boolean
        if _AFFIRMATIVE.match(text):
            slots.append(Slot("boolean", True, text, 0.95))
        elif _NEGATIVE.match(text):
            slots.append(Slot("boolean", False, text, 0.95))

        # Years of experience
        m = _YEARS_RE.search(text)
        if m:
            years = float(m.group(1))
            slots.append(
                Slot(
                    "years_experience",
                    years,
                    m.group(0),
                    0.9,
                )
            )
        elif _FRESHER_RE.search(text):
            slots.append(
                Slot(
                    "years_experience",
                    0.0,
                    text,
                    0.85,
                )
            )

        # Availability / notice period
        if _IMMEDIATE_RE.search(text):
            slots.append(
                Slot(
                    "availability",
                    "immediate",
                    text,
                    0.9,
                )
            )
        else:
            d = _DURATION_RE.search(text)
            if d:
                value = _normalize_duration(
                    d.group(1),
                    d.group(2),
                )
                slots.append(
                    Slot(
                        "notice_period_days",
                        value,
                        d.group(0),
                        0.9,
                    )
                )

        # Join-by date
        date_match = _DATE_RE.search(text)
        if date_match:
            slots.append(
                Slot(
                    "join_by_date",
                    date_match.group(0),
                    date_match.group(0),
                    0.7,
                )
            )

        # Money / salary
        m = _MONEY_LPA_RE.search(text)
        if m:
            lakhs = float(m.group(1))
            slots.append(
                Slot(
                    "expected_salary_lakhs",
                    lakhs,
                    m.group(0),
                    0.9,
                )
            )
        else:
            m = _MONEY_K_RE.search(text)
            if m:
                thousands = float(m.group(1))
                lakhs = thousands / 100.0
                slots.append(
                    Slot(
                        "expected_salary_lakhs",
                        lakhs,
                        m.group(0),
                        0.7,
                    )
                )
            else:
                m = _MONEY_FULL_RE.search(text)
                if m:
                    raw = m.group(1).replace(",", "")
                    value_inr = _normalize_money_to_lakhs(raw)

                    if value_inr:
                        slots.append(
                            Slot(
                                "expected_salary_lakhs",
                                value_inr,
                                m.group(0),
                                0.6,
                            )
                        )

        # Skills
        lower = text.lower()

        found_skills = sorted(
            {
                skill
                for skill in _SKILL_VOCAB
                if re.search(
                    rf"\b{re.escape(skill)}\b",
                    lower,
                )
            }
        )

        if found_skills:
            slots.append(
                Slot(
                    "skills",
                    found_skills,
                    text,
                    0.85,
                )
            )

        # City
        for city in sorted(
            _CITIES,
            key=len,
            reverse=True,
        ):
            if re.search(
                rf"\b{re.escape(city)}\b",
                lower,
            ):
                canonical = _CITY_ALIASES[city]

                slots.append(
                    Slot(
                        "city",
                        canonical,
                        city,
                        0.9,
                    )
                )
                break

        # Optional LLM pass
        if self.use_llm:
            try:
                llm_slots = self._call_llm(text)
                slots.extend(llm_slots)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "LLM slot extraction failed: %s",
                    exc,
                )

        return slots

    def _assess_quality(
        self,
        text: str,
        intent: IntentResult,
        slots: List[Slot],
        expected_slot: Optional[str],
    ) -> Dict[str, Any]:
        """
        Combine intent + slot presence into quality flags
        and a 0-1 completeness score.
        """
        warnings: List[str] = []

        is_off_topic = intent.label == "off_topic"

        is_vague = bool(_VAGUE_MARKERS.search(text)) and not is_off_topic

        is_missing = intent.label == "no_response" or not text

        slot_names = {s.name for s in slots}

        # Required-slot check
        if expected_slot and expected_slot not in slot_names and not is_missing:
            is_missing = True
            warnings.append(f"Expected slot '{expected_slot}' was not extracted.")

        # Off-topic or no response
        if is_off_topic or intent.label == "no_response":
            completeness = 0.0

        elif intent.label == "objection":
            completeness = 0.2
            warnings.append("Candidate objected; consider rephrasing or escalating.")

        elif intent.label == "redirect":
            completeness = 0.1

        elif intent.label == "clarification_request":
            completeness = 0.2

        else:
            completeness = 1.0 if slot_names else 0.3

            if is_vague:
                completeness *= 0.5
                warnings.append("Reply uses vague qualifiers " "(maybe / kind of / etc.).")

        if intent.confidence < 0.6:
            warnings.append(f"Low intent confidence ({intent.confidence:.2f}).")

        return {
            "is_off_topic": is_off_topic,
            "is_vague": is_vague,
            "is_missing": is_missing,
            "completeness": max(
                0.0,
                min(1.0, completeness),
            ),
            "warnings": warnings,
        }

    def _call_llm(self, text: str) -> List[Slot]:
        """
        Optional LLM-based slot extraction.

        Mirrors the screening_ai.screener._call_llm pattern so
        tests can monkeypatch this method.
        """
        from openai import OpenAI

        from config.settings import settings

        client = OpenAI(api_key=settings.openai_api_key)

        client.chat.completions.create(
            model=settings.ai_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract structured slots from a "
                        "candidate's screening answer. Return JSON: "
                        '{"slots":[{"name":..., "value":..., '
                        '"confidence":...}]}'
                    ),
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
        )

        # Scaffold: real implementation would parse JSON.
        # Tests can patch this method.
        return []


# --- small helpers --------------------------------------------------------- #


def _normalize_duration(
    num_str: str,
    unit_str: str,
) -> int:
    """Return the duration as days."""
    n = int(num_str)
    unit = unit_str.lower()

    if unit.startswith("day"):
        return n

    if unit.startswith("week"):
        return n * 7

    if unit.startswith("month"):
        return n * 30

    return n


def _normalize_money_to_lakhs(
    raw: str,
) -> Optional[float]:
    """
    Best-effort conversion of a numeric amount to INR lakhs.

    Returns None if the number is invalid.
    """
    try:
        n = float(raw)
    except ValueError:
        return None

    # Heuristic: anything under 1000 is treated as lakhs.
    if n < 1000:
        return n

    return round(n / 100000.0, 2)
