"""
answer_understanding_engine/intent_classifier.py
-------------------------------------------------
Day 25 — Intent Classifier (standalone module).

A dedicated, focused module that takes a candidate's raw answer text
and returns an `IntentResult` describing what the reply is *doing*.

Six intent labels:
    - answer                 on-topic reply to the question
    - clarification_request  reply is itself a question
    - objection              candidate raised a concern
    - redirect               candidate tried to change the subject
    - off_topic              clearly unrelated reply
    - no_response            empty / filler / "I don't know"

Public API:
    from answer_understanding_engine.intent_classifier import (
        classify_intent,
        IntentResult,
        INTENT_LABELS,
    )

    result = classify_intent("I have 5 years of experience.")
    print(result.label)        # "answer"
    print(result.confidence)   # 0.8
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

# --- imports from shared utilities --------------------------------------- #
from utils.logger import get_logger

logger = get_logger("answer_understanding_engine.intent_classifier")


# --------------------------------------------------------------------------- #
# Public dataclass                                                            #
# --------------------------------------------------------------------------- #


@dataclass
class IntentResult:
    """Classification of what the candidate's reply is *doing*."""

    label: str  # one of INTENT_LABELS
    confidence: float  # 0.0 – 1.0
    rationale: str  # short human-readable explanation

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Labels                                                                      #
# --------------------------------------------------------------------------- #

INTENT_LABELS: List[str] = [
    "answer",
    "clarification_request",
    "objection",
    "redirect",
    "off_topic",
    "no_response",
]


# --------------------------------------------------------------------------- #
# Pattern tables (public, so callers can inspect or override)                #
# --------------------------------------------------------------------------- #

# A reply whose only "content" is filler counts as no_response.
NO_RESPONSE_PATTERNS: List[str] = [
    r"^\s*$",
    r"^(idk|i don'?t know|no idea|not sure|na|n/?a|none|nothing)\b",
    r"^(uh+|um+|hmm+|er+|ah+)\s*[.,!]?\s*$",
]

# Off-topic indicators — used after an on-topic pre-check to detect a
# clear non-sequitur (recipe, YouTube plug, sports score, ...).
OFF_TOPIC_RE = re.compile(
    r"\b("
    r"recipe|cricket score|movie review|weather forecast|"
    r"bitcoin price|stock tip|joke|poem|love letter|"
    r"buy my course|promote my (channel|product)|"
    r"(youtube|you.?tube) channel"
    r")\b",
    re.IGNORECASE,
)

# Question-marker hints (clarification_request vs answer).
QUESTION_MARKERS = re.compile(
    r"\?$|^(what|why|how|when|where|who|can you|could you)\b",
    re.IGNORECASE,
)

# "I want to talk about X instead" patterns.
REDIRECT_PATTERNS = re.compile(
    r"\b(instead|rather (talk|discuss)|can we discuss|let'?s talk about|"
    r"skip|change the subject|move on|next question)\b",
    re.IGNORECASE,
)

# "I have a problem with …" patterns.
OBJECTION_MARKERS = re.compile(
    r"\b(i (don'?t|do not) (want|like|feel)|"
    r"this (doesn'?t|does not) (feel|seem) right|"
    r"i'?m (not interested|uncomfortable|busy)|"
    r"why (should i|do i)|"
    r"this is (a waste|unfair|biased))\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Classifier                                                                  #
# --------------------------------------------------------------------------- #


def classify_intent(text: str) -> IntentResult:
    """
    Classify the intent of a candidate's free-form reply.

    Parameters
    ----------
    text : str
        The raw candidate reply. Leading/trailing whitespace ignored.

    Returns
    -------
    IntentResult
        A result with `label`, `confidence`, and `rationale`.

    Examples
    --------
    >>> classify_intent("").label
    'no_response'
    >>> classify_intent("I have 5 years experience.").label
    'answer'
    >>> classify_intent("What do you mean?").label
    'clarification_request'
    >>> classify_intent("Here's a recipe for biryani.").label
    'off_topic'
    """
    text = (text or "").strip()
    logger.debug("Classifying intent for answer of length %d", len(text))

    # 1. No response — check first, since empty/filler short-circuits everything else
    if not text:
        return IntentResult("no_response", 0.99, "Empty reply.")
    for pat in NO_RESPONSE_PATTERNS:
        if re.match(pat, text, re.IGNORECASE):
            return IntentResult("no_response", 0.95, "Filler or explicit no-info reply.")

    # 2. Off-topic — explicit non-sequitur keywords
    if OFF_TOPIC_RE.search(text):
        return IntentResult("off_topic", 0.9, "Reply contains an off-topic keyword.")

    # 3. Redirect — candidate tries to change the subject
    if REDIRECT_PATTERNS.search(text):
        return IntentResult("redirect", 0.75, "Candidate tried to change the subject.")

    # 4. Objection — candidate raised a concern
    if OBJECTION_MARKERS.search(text):
        return IntentResult("objection", 0.75, "Candidate raised a concern / objection.")

    # 5. Clarification request — the reply is itself a question
    if QUESTION_MARKERS.search(text):
        return IntentResult("clarification_request", 0.7, "Reply is itself a question.")

    # 6. Default — on-topic answer
    return IntentResult("answer", 0.8, "Treating reply as an on-topic answer.")


__all__ = [
    "IntentResult",
    "INTENT_LABELS",
    "classify_intent",
    "NO_RESPONSE_PATTERNS",
    "OFF_TOPIC_RE",
    "QUESTION_MARKERS",
    "REDIRECT_PATTERNS",
    "OBJECTION_MARKERS",
]
