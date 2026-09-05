"""
dimension_scorers.py
--------------------
Day 26 deliverable — Zecpath AI Job Portal

The four orthogonal scoring dimensions, each implemented as a pure
function. Each function takes a candidate's raw answer plus the
question spec and returns a (score, confidence, notes) tuple.

All scorers are designed to be:
  - Deterministic (no LLM call required, all rule-based)
  - Testable (pure functions, no I/O, no globals)
  - Additive (they read AnswerUnderstandingEngine.StructuredAnswer but
    do not depend on it; can be called with the raw answer alone)

The four dimensions:
  1. Clarity         -- how clear / well-formed is the reply?
  2. Relevance       -- does the reply actually answer the question?
  3. Completeness    -- does the reply contain all expected information?
  4. Consistency     -- does the reply agree with other answers in the session?
                        (lives in consistency.py because it needs cross-question context)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ---- public dataclass ---- #


@dataclass
class DimensionResult:
    score: float
    confidence: float
    notes: List[str]


# ---- shared text helpers ---- #

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#.]{1,}")
_VAGUE_MARKERS = re.compile(
    r"\b(maybe|kind of|sort of|i think|i guess|it depends|approximately|"
    r"roughly|around|probably|perhaps|might be)\b",
    re.IGNORECASE,
)
_AFFIRMATIVE = re.compile(
    r"^\s*(yes|yeah|yep|sure|of course|definitely|absolutely|"
    r"i (can|do|am|will)|happy to|no problem|ok(ay)?)\b",
    re.IGNORECASE,
)
_NEGATIVE = re.compile(
    r"^\s*(no|nope|nah|not really|i (can'?t|don'?t|won'?t|am not)|"
    r"unfortunately|afraid not|never)\b",
    re.IGNORECASE,
)
_FILLERS = re.compile(
    r"\b(um+|uh+|er+|ah+|hmm+|like|you know|i mean|kind of)\b",
    re.IGNORECASE,
)


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def _has_terminal_punctuation(text: str) -> bool:
    return bool(text and text.rstrip()[-1] in ".!?")


def _ends_with_affirmation_or_negation(text: str) -> bool:
    return bool(_AFFIRMATIVE.match(text or "")) or bool(_NEGATIVE.match(text or ""))


# ---- 1. CLARITY ---- #


def score_clarity(
    raw_answer: str,
    expected_answer_type: str,
    *,
    min_words_for_text: int = 3,
) -> DimensionResult:
    """
    Measure how clear and well-formed the reply is.

    Penalizes:
      - Empty / very short replies
      - Vague qualifiers ("maybe", "kind of", "I think")
      - Excessive filler words
      - Run-on / rambling (> 80 words for a short screening answer is unusual)
      - Missing terminal punctuation (signals an unfinished thought)

    A "no_response" or empty string short-circuits to 0.0.
    """
    notes: List[str] = []
    text = (raw_answer or "").strip()

    if not text:
        return DimensionResult(0.0, 0.99, ["Empty reply — no clarity to assess."])

    score = 1.0
    word_count = _word_count(text)

    # Boolean answers: short "yes"/"no" is fine; don't penalize for length.
    if expected_answer_type == "boolean":
        if word_count > 6:
            score -= 0.15
            notes.append(f"Verbose boolean answer ({word_count} words).")
    else:
        # Length sweet spot: between 4 and 60 words is typical for screening answers.
        if word_count < 2:
            score -= 0.4
            notes.append(f"Very short reply ({word_count} words).")
        elif word_count < min_words_for_text and expected_answer_type == "text":
            score -= 0.2
            notes.append(f"Short reply ({word_count} words) for an open-text question.")

    if word_count > 80:
        score -= 0.15
        notes.append(f"Long reply ({word_count} words); possible rambling.")

    # Vague qualifiers hurt clarity.
    vague_hits = _VAGUE_MARKERS.findall(text)
    if vague_hits:
        score -= min(0.4, 0.1 * len(vague_hits))
        notes.append(f"Vague qualifier(s) detected: {sorted(set(vague_hits))}.")

    # Filler words.
    filler_hits = _FILLERS.findall(text)
    filler_ratio = len(filler_hits) / max(word_count, 1)
    if filler_ratio > 0.1:
        score -= min(0.3, filler_ratio)
        notes.append(f"Filler-word ratio {filler_ratio:.0%}.")

    # Missing terminal punctuation (signals a truncated reply).
    if not _has_terminal_punctuation(text) and expected_answer_type == "text" and word_count > 3:
        score -= 0.1
        notes.append("No terminal punctuation; possible truncation.")

    score = max(0.0, min(1.0, score))

    # Confidence: more words = more signal.
    if word_count >= 4:
        confidence = 0.85
    elif word_count >= 2:
        confidence = 0.6
    else:
        confidence = 0.4

    return DimensionResult(score=score, confidence=confidence, notes=notes)


# ---- 2. RELEVANCE ---- #

# Category-level expected slot hints. Used to decide whether the slots
# extracted by AnswerUnderstandingEngine actually answer the question.
_CATEGORY_EXPECTED_SLOT = {
    "introduction": ("boolean", "text"),
    "education": ("text", "boolean"),
    "experience": ("text", "number", "duration"),
    "skills": ("text", "number", "boolean"),
    "location": ("text", "boolean"),
    "salary": ("number", "boolean", "text"),
    "notice_period": ("duration", "number", "date", "boolean"),
}


# Keywords per category, used as a soft relevance signal when no
# extracted slots are available.
_CATEGORY_KEYWORDS = {
    "experience": [
        "year",
        "years",
        "experience",
        "worked",
        "company",
        "engineer",
        "developer",
        "manager",
        "lead",
    ],
    "education": [
        "degree",
        "bachelor",
        "master",
        "phd",
        "b.tech",
        "btech",
        "be",
        "mba",
        "university",
        "college",
        "institute",
        "graduate",
        "diploma",
    ],
    "skills": [
        "python",
        "java",
        "javascript",
        "aws",
        "react",
        "node",
        "sql",
        "kubernetes",
        "docker",
        "machine learning",
        "communication",
        "negotiation",
        "design",
        "seo",
        "analytics",
    ],
    "location": [
        "mumbai",
        "delhi",
        "bengaluru",
        "bangalore",
        "hyderabad",
        "chennai",
        "pune",
        "kolkata",
        "remote",
        "relocate",
        "relocation",
        "willing",
        "based",
    ],
    "salary": [
        "lakh",
        "lakhs",
        "lpa",
        "rs",
        "inr",
        "₹",
        "salary",
        "compensation",
        "ctc",
        "negotiable",
        "expected",
        "current",
    ],
    "notice_period": [
        "immediate",
        "notice",
        "month",
        "week",
        "day",
        "serving",
        "buyout",
        "reduce",
        "join",
        "asap",
        "today",
    ],
    "introduction": [
        "name",
        "applied",
        "interview",
        "candidate",
    ],
}


def score_relevance(
    raw_answer: str,
    expected_answer_type: str,
    category: str,
    extracted: Optional[Dict[str, Any]] = None,
    intent_label: Optional[str] = None,
) -> DimensionResult:
    """
    Measure whether the reply actually answers the question.

    Heuristics:
      - If intent_label is "off_topic" / "no_response" / "redirect", score 0.
      - If the expected answer type is "boolean" and the reply resolves to
        yes/no, score 1.0; otherwise 0.4 (didn't answer as expected).
      - If the expected answer type is "number" and the reply contains a
        number, score high.
      - If the reply contains category-relevant keywords OR extracted
        slots, score 1.0; else 0.4.
      - Otherwise, partial credit (0.5-0.7) for a plausible but vague reply.
    """
    notes: List[str] = []
    text = (raw_answer or "").strip()
    extracted = extracted or {}

    # Off-topic / no-response intents are scoring killers.
    if intent_label in ("off_topic", "no_response"):
        return DimensionResult(
            score=0.0,
            confidence=0.95,
            notes=[f"Intent was '{intent_label}'; reply not relevant."],
        )
    if intent_label == "redirect":
        return DimensionResult(
            score=0.1,
            confidence=0.7,
            notes=["Candidate redirected the topic; not directly relevant."],
        )
    if intent_label == "objection":
        return DimensionResult(
            score=0.3,
            confidence=0.6,
            notes=["Candidate raised an objection; not a direct answer."],
        )

    if not text:
        return DimensionResult(0.0, 0.99, ["Empty reply; not relevant."])

    # Type-specific checks first.
    if expected_answer_type == "boolean":
        if _ends_with_affirmation_or_negation(text):
            score = 1.0
            notes.append("Boolean question answered with yes/no.")
        else:
            score = 0.4
            notes.append("Boolean question; did not detect yes/no.")
        return DimensionResult(score=score, confidence=0.9, notes=notes)

    if expected_answer_type == "number":
        if any(ch.isdigit() for ch in text):
            score = 0.95
            notes.append("Number detected in reply.")
        else:
            score = 0.3
            notes.append("Number expected; none found in reply.")
        return DimensionResult(score=score, confidence=0.85, notes=notes)

    if expected_answer_type == "duration":
        if _DURATION_RE.search(text) or any(ch.isdigit() for ch in text):
            score = 0.95
            notes.append("Duration/numeric value detected.")
        else:
            score = 0.3
            notes.append("Duration expected; not detected.")
        return DimensionResult(score=score, confidence=0.8, notes=notes)

    if expected_answer_type == "date":
        if _DATE_RE.search(text):
            score = 0.95
            notes.append("Date detected in reply.")
        else:
            score = 0.4
            notes.append("Date expected; not clearly detected.")
        return DimensionResult(score=score, confidence=0.7, notes=notes)

    # text/enum fallback: rely on extracted slots + keyword matching.
    cat = (category or "").lower()
    expected_slots = _CATEGORY_EXPECTED_SLOT.get(cat, ())
    slot_match = any(name in extracted for name in expected_slots) if expected_slots else False

    keywords = _CATEGORY_KEYWORDS.get(cat, [])
    lowered = text.lower()
    keyword_hits = [k for k in keywords if k in lowered]

    if slot_match and keyword_hits:
        score = 1.0
        notes.append(f"Extracted slot(s) + {len(keyword_hits)} category keyword hit(s).")
    elif slot_match:
        score = 0.85
        notes.append("Extracted slot matches expected category.")
    elif keyword_hits:
        score = 0.8
        notes.append(f"{len(keyword_hits)} category keyword hit(s): {keyword_hits[:3]}.")
    else:
        # No strong signal. Give partial credit for any non-trivial reply.
        wc = _word_count(text)
        if wc >= 5:
            score = 0.55
            notes.append("No category signal; partial credit for non-trivial reply.")
        elif wc >= 2:
            score = 0.4
            notes.append("No category signal; reply too short to assess relevance.")
        else:
            score = 0.2
            notes.append("No category signal; reply too short to be relevant.")

    return DimensionResult(score=score, confidence=0.7, notes=notes)


# ---- 3. COMPLETENESS ---- #


def score_completeness(
    raw_answer: str,
    expected_answer_type: str,
    expected_slot: Optional[str] = None,
    extracted: Optional[Dict[str, Any]] = None,
    intent_label: Optional[str] = None,
) -> DimensionResult:
    """
    Measure whether the reply contains all expected information.

    If `expected_slot` is given (e.g. "years_experience", "expected_salary"),
    the dimension wants to see a Slot of that name. Otherwise completeness
    is approximated by:
      - boolean question: must resolve to yes/no
      - number question: must contain a number
      - duration: must contain a numeric duration
      - text: must be non-trivial (>= 3 words)
    """
    notes: List[str] = []
    text = (raw_answer or "").strip()
    extracted = extracted or {}

    if intent_label in ("no_response", "off_topic"):
        return DimensionResult(0.0, 0.95, ["Reply not substantive."])

    if not text:
        return DimensionResult(0.0, 0.99, ["Empty reply; nothing to assess."])

    # If the call-site provided an explicit expected slot, check for it.
    if expected_slot:
        if expected_slot in extracted:
            return DimensionResult(
                score=1.0,
                confidence=0.9,
                notes=[f"Expected slot '{expected_slot}' was extracted."],
            )
        # Slot missing.
        # Give partial credit if the answer is otherwise non-trivial.
        wc = _word_count(text)
        partial = min(0.5, 0.1 * wc)
        return DimensionResult(
            score=partial,
            confidence=0.7,
            notes=[f"Expected slot '{expected_slot}' missing from reply."],
        )

    # Type-driven completeness.
    if expected_answer_type == "boolean":
        ok = _ends_with_affirmation_or_negation(text)
        return DimensionResult(
            score=1.0 if ok else 0.3,
            confidence=0.9,
            notes=["Boolean resolved" if ok else "Boolean not clearly resolved."],
        )

    if expected_answer_type == "number":
        ok = any(ch.isdigit() for ch in text)
        return DimensionResult(
            score=1.0 if ok else 0.3,
            confidence=0.85,
            notes=["Number present" if ok else "Number missing."],
        )

    if expected_answer_type == "duration":
        ok = bool(_DURATION_RE.search(text) or any(ch.isdigit() for ch in text))
        return DimensionResult(
            score=1.0 if ok else 0.3,
            confidence=0.8,
            notes=["Duration present" if ok else "Duration missing."],
        )

    if expected_answer_type == "date":
        ok = bool(_DATE_RE.search(text))
        return DimensionResult(
            score=1.0 if ok else 0.3,
            confidence=0.75,
            notes=["Date present" if ok else "Date missing."],
        )

    # text / enum: based on word count and any extracted slot presence.
    wc = _word_count(text)
    if wc >= 8:
        score = 1.0
        notes.append(f"Substantive reply ({wc} words).")
    elif wc >= 4:
        score = 0.75
        notes.append(f"Moderate reply ({wc} words).")
    elif wc >= 2:
        score = 0.5
        notes.append(f"Short reply ({wc} words).")
    else:
        score = 0.2
        notes.append("Reply is one word; low completeness.")

    if extracted:
        score = min(1.0, score + 0.05)
        notes.append("Bonus: at least one slot extracted.")

    return DimensionResult(score=score, confidence=0.7, notes=notes)


# ---- Shared regex (imported by relevance + consistency) ---- #

_DURATION_RE = re.compile(
    r"\b(\d+)\s*(day|week|month|months|days|weeks)s?\b",
    re.IGNORECASE,
)

_DATE_RE = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"[a-z]*\s+\d{2,4}|"
    r"(next|this)\s+(week|month|monday))\b",
    re.IGNORECASE,
)
