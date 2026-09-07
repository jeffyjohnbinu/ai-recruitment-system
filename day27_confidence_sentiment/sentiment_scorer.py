"""
sentiment_scorer.py
-------------------
Day 27 deliverable — Zecpath AI Job Portal

Rule-based sentiment analysis for interview answers.

Detects:
  1. Polarity — positive, negative, neutral, or mixed
  2. Emotion signals — enthusiasm, frustration, confidence, anxiety, calm
  3. Self-contradictions — candidate says one thing, then the opposite

All functions are pure and deterministic. LLM stubs are prefixed `_llm_`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .formats import SentimentScore

# ---- Lexicons ---- #

# Strongly positive words/phrases.
_POSITIVE_WORDS = {
    "excellent",
    "outstanding",
    "exceptional",
    "amazing",
    "fantastic",
    "great",
    "wonderful",
    "brilliant",
    "superb",
    "perfect",
    "love",
    "loved",
    "enjoy",
    "enjoyed",
    "excited",
    "exciting",
    "passionate",
    "passion",
    "enthusiastic",
    "enthusiasm",
    "confident",
    "confidently",
    "certain",
    "definitely",
    "absolutely",
    "sure",
    "certainly",
    "obviously",
    "clearly",
    "obviously",
    "happy",
    "glad",
    "pleased",
    "delighted",
    "thrilled",
    "proud",
    "achieved",
    "accomplished",
    "successful",
    "success",
    "best",
    "better",
    "improved",
    "improvement",
    "growth",
    "solved",
    "solution",
    "fixed",
    "resolved",
    "handled",
    "lead",
    "led",
    "managed",
    "mentored",
    "trained",
    "increase",
    "increased",
    "reduce",
    "reduced",
    "saved",
    "automate",
    "automated",
    "optimize",
    "optimized",
    "strong",
    "strongest",
    "expertise",
    "expert",
    "skilled",
    "recommend",
    "commended",
    "awarded",
    "promoted",
    "yes",
    "yeah",
    "yep",
    "definitely",
    "absolutely",
    "of course",
    "i can",
    "i will",
    "i did",
    "i have",
    "i'm able",
    "no problem",
    "happy to",
    "glad to",
    "pleased to",
}

# Strongly negative words/phrases.
_NEGATIVE_WORDS = {
    "bad",
    "poor",
    "terrible",
    "awful",
    "horrible",
    "dreadful",
    "hate",
    "hated",
    "dislike",
    "disliked",
    "disappointed",
    "disappointing",
    "frustrated",
    "frustrating",
    "frustration",
    "annoyed",
    "annoying",
    "angry",
    "upset",
    "irritated",
    "irritating",
    "failed",
    "failure",
    "fail",
    "failing",
    "wrong",
    "mistake",
    "mistaken",
    "problem",
    "problems",
    "issue",
    "issues",
    "difficult",
    "difficulty",
    "hard",
    "harder",
    "hardest",
    "challenging",
    "challenge",
    "weak",
    "weakness",
    "weaknesses",
    "inadequate",
    "insufficient",
    "lacked",
    "lacking",
    "missing",
    "absent",
    "nonexistent",
    "never",
    "nothing",
    "nobody",
    "nowhere",
    "no one",
    "cannot",
    "can't",
    "couldn't",
    "won't",
    "wouldn't",
    "shouldn't",
    "no",
    "nope",
    "nah",
    "not",
    "negative",
    "i don't know",
    "i'm not sure",
    "i'm not certain",
    "i don't remember",
    "i don't recall",
    "worst",
    "worse",
    "declined",
    "decline",
    "decreased",
    "decrease",
    "lost",
    "lose",
    "losing",
    "waste",
    "wasted",
    "useless",
    "boring",
    "bored",
    "tired",
    "exhausted",
    "overwhelmed",
    "stressed",
    "stressful",
    "anxious",
    "worried",
    "concerned",
}

# Neutral / factual words (used to adjust confidence).
_NEUTRAL_WORDS = {
    "worked",
    "working",
    "work",
    "job",
    "role",
    "position",
    "company",
    "team",
    "teamwork",
    "colleague",
    "colleagues",
    "project",
    "projects",
    "task",
    "tasks",
    "assignment",
    "assignments",
    "responsible",
    "responsibility",
    "responsible for",
    "experience",
    "experienced",
    "year",
    "years",
    "degree",
    "education",
    "qualification",
    "certified",
    "certification",
    "skill",
    "skills",
    "tool",
    "tools",
    "technology",
    "technologies",
    "used",
    "using",
    "utilize",
    "utilized",
    "developed",
    "developing",
    "created",
    "creating",
    "managed",
    "managing",
    "led",
    "leading",
    "implemented",
    "implementing",
    "deployed",
    "deploying",
    "reviewed",
    "reviewing",
    "analyzed",
    "analyzing",
    "report",
    "reports",
    "document",
    "documentation",
    "meet",
    "meeting",
    "meetings",
    "standup",
    "stand-up",
    "deadline",
    "deadlines",
    "sprint",
    "sprints",
    "requirement",
    "requirements",
    "spec",
    "specification",
}

# Emotion signal keywords.
_EMOTION_KEYWORDS = {
    "enthusiasm": {
        "excited",
        "exciting",
        "passionate",
        "love",
        "enjoy",
        "thrilled",
        "eager",
        "motivated",
        "energetic",
        "fired up",
        "great",
        "fantastic",
        "wonderful",
    },
    "frustration": {
        "frustrated",
        "frustrating",
        "annoying",
        "annoyed",
        "difficult",
        "hard",
        "problem",
        "issue",
        "bug",
        "slow",
        "delayed",
        "blocked",
        "stuck",
    },
    "confidence": {
        "confident",
        "certain",
        "sure",
        "definitely",
        "absolutely",
        "obviously",
        "clearly",
        "i know",
        "i'm sure",
        "i can",
        "no problem",
        "handled",
        "solved",
        "resolved",
    },
    "anxiety": {
        "anxious",
        "nervous",
        "worried",
        "concerned",
        "stressed",
        "overwhelmed",
        "scared",
        "fear",
        "uncertain",
        "unsure",
        "doubt",
        "doubtful",
    },
    "calm": {
        "calm",
        "relaxed",
        "composed",
        "steady",
        "balanced",
        "steady",
        "measured",
        "careful",
        "thoughtful",
    },
    "pride": {
        "proud",
        "achieved",
        "accomplished",
        "successful",
        "success",
        "best",
        "won",
        "awarded",
        "promoted",
        "commended",
    },
}

# Negation context window (number of words to look back).
_NEGATION_WINDOW = 4


def _count_word_matches(text: str, word_set: set) -> int:
    """Count how many words from the set appear in the text."""
    lowered = text.lower()
    words = set(re.findall(r"[a-z]+", lowered))
    return len(words & word_set)


def _detect_negation_context(text: str) -> List[int]:
    """
    Return starting character indices of all negation words in text.
    """
    return [m.start() for m in re.finditer(_NEGATION_RE, text)]


# Negation regex (re-declared here to avoid circular imports).
_NEGATION_RE = re.compile(
    r"\b(not|never|no|none|nothing|nowhere|neither|nobody|hardly|barely|rarely)\b",
    re.IGNORECASE,
)


@dataclass
class SentimentResult:
    """Internal sentiment analysis result."""

    polarity: float
    positive_score: float
    negative_score: float
    neutral_score: float
    sentiment_label: str
    confidence: float
    emotion_signals: Dict[str, float]
    positive_words_found: List[str]
    negative_words_found: List[str]


def classify_polarity(
    raw_answer: str,
    negation_context: bool = True,
) -> Tuple[float, float, float]:
    """
    Compute positive, negative, and neutral scores for an answer.

    Returns (positive_score, negative_score, neutral_score) — each 0.0-1.0.

    With negation_context=True, words following a negation ("not good")
    flip from positive to negative (or vice versa).
    """
    text = raw_answer or ""
    lowered = text.lower()

    # Find negation positions.
    negation_positions = _detect_negation_context(text)
    negation_ranges = [
        (pos, pos + 20) for pos in negation_positions  # 20-char window after negation
    ]

    def _in_negation_context(char_idx: int) -> bool:
        return any(start <= char_idx < end for start, end in negation_ranges)

    # Count positive and negative words, respecting negation context.
    words = re.findall(r"[a-z]+", lowered)
    pos_found: List[str] = []
    neg_found: List[str] = []

    # We need character positions. Build a simple word -> char-index map.
    word_positions: List[Tuple[str, int]] = []
    for m in re.finditer(r"[a-z]+", lowered):
        word_positions.append((m.group(), m.start()))

    total_weighted = 0.0
    pos_weighted = 0.0
    neg_weighted = 0.0

    for word, char_idx in word_positions:
        in_neg = _in_negation_context(char_idx)
        word_in_pos = word in _POSITIVE_WORDS
        word_in_neg = word in _NEGATIVE_WORDS
        word_in_neutral = word in _NEUTRAL_WORDS

        if word_in_pos or word_in_neg or word_in_neutral:
            weight = 1.0
            if word in _POSITIVE_WORDS:
                pos_found.append(word)
            elif word in _NEGATIVE_WORDS:
                neg_found.append(word)

            # Negation flips polarity.
            if in_neg and word_in_pos:
                neg_weighted += weight
            elif in_neg and word_in_neg:
                pos_weighted += weight * 0.7  # soften double-negatives
            elif word_in_pos:
                pos_weighted += weight
            elif word_in_neg:
                neg_weighted += weight
            else:
                # Neutral word: contributes to denominator.
                total_weighted += weight

    # Normalize.
    total_signal = pos_weighted + neg_weighted + 0.1  # epsilon to avoid div/0
    pos_score = pos_weighted / total_signal
    neg_score = neg_weighted / total_signal
    neutral_score = 1.0 - (pos_score + neg_score)
    neutral_score = max(0.0, min(1.0, neutral_score))

    return pos_score, neg_score, neutral_score


def _compute_polarity(pos_score: float, neg_score: float) -> float:
    """Map pos/neg scores to a single polarity score: -1 to +1."""
    return pos_score - neg_score


def _label_sentiment(polarity: float, pos_score: float, neg_score: float) -> str:
    if pos_score > 0.5 and neg_score < 0.15:
        return "positive"
    elif neg_score > 0.5 and pos_score < 0.15:
        return "negative"
    elif pos_score > 0.3 and neg_score > 0.3:
        return "mixed"
    return "neutral"


def _emotion_signal_score(emotion_words: set, text: str) -> float:
    """Return a 0-1 signal score for one emotion based on keyword frequency."""
    if not text.strip():
        return 0.0
    matches = _count_word_matches(text, emotion_words)
    # Diminishing returns: 1→0.4, 2→0.7, 3→0.85, 4+→1.0
    if matches == 0:
        return 0.0
    return min(1.0, 0.3 + 0.2 * matches)


def score_sentiment(
    raw_answer: str,
    category: Optional[str] = None,
) -> SentimentScore:
    """
    Compute sentiment polarity and emotion signals for an answer.

    Args:
        raw_answer: The candidate's answer text.
        category: Optional category hint for category-specific scoring adjustments.

    Returns:
        SentimentScore with polarity, component scores, emotion signals, and label.
    """
    text = raw_answer or ""
    if not text.strip():
        return SentimentScore(
            polarity=0.0,
            positive_score=0.0,
            negative_score=0.0,
            neutral_score=1.0,
            sentiment_label="neutral",
            confidence=1.0,
            emotion_signals={},
            note="Empty answer; neutral sentiment.",
        )

    # Pos/neg/neutral scores.
    pos_score, neg_score, neutral_score = classify_polarity(text)

    # Polarity.
    polarity = _compute_polarity(pos_score, neg_score)

    # Sentiment label.
    label = _label_sentiment(polarity, pos_score, neg_score)

    # Confidence in the sentiment assessment.
    word_count = len(re.findall(r"[a-z]+", text.lower()))
    confidence = min(1.0, word_count / 10.0)  # More words = higher confidence.

    # Detect positive/negative words found.
    pos_found = [w for w in _POSITIVE_WORDS if w in text.lower()]
    neg_found = [w for w in _NEGATIVE_WORDS if w in text.lower()]

    # Emotion signals.
    emotion_signals: Dict[str, float] = {}
    for emotion, keywords in _EMOTION_KEYWORDS.items():
        signal_score = _emotion_signal_score(keywords, text)
        if signal_score > 0:
            emotion_signals[emotion] = signal_score

    # Category-specific adjustments.
    note_parts: List[str] = []
    if category == "experience" and neg_found:
        # Some negativity in experience descriptions is normal.
        note_parts.append("Mild negative sentiment in experience context is expected.")
    if category == "salary" and neg_found:
        note_parts.append("Negative sentiment around salary expectations warrants review.")

    return SentimentScore(
        polarity=round(polarity, 3),
        positive_score=round(pos_score, 3),
        negative_score=round(neg_score, 3),
        neutral_score=round(neutral_score, 3),
        sentiment_label=label,
        confidence=round(confidence, 3),
        emotion_signals={k: round(v, 3) for k, v in emotion_signals.items()},
        note=" ".join(note_parts),
    )


# ---- Self-contradiction detection ---- #

# Common contradictory word pairs.
_CONTRADICTION_PAIRS = [
    # Certainty vs. uncertainty.
    (
        r"\b(i am|i'm|am) (confident|sure|certain)\b",
        r"\b(not|never|don't)\b.*\b(know|remember|recall)\b",
    ),
    (r"\bdefinitely\b", r"\bmaybe\b"),
    (r"\babsolutely\b", r"\bmight not\b"),
    # Past vs. present contradiction.
    (r"\b(i have|i had|i've)\b.*\b(experience|worked|managed)\b", r"\bi don't (have|have no)\b"),
    # Ability contradiction.
    (r"\b(i can|i am able to|i will)\b", r"\b(i cannot|i can't|i could not|i couldn't)\b"),
    # Commitment contradiction.
    (
        r"\b(i am|i'm)\b.*\b(committed|dedicated|motivated)\b",
        r"\bnot (interested|motivated|committed)\b",
    ),
    # Success vs. failure.
    (r"\bsuccess(s|ful)?\b", r"\bfail(ed|ure)?\b"),
    (r"\bachieved?\b", r"\bfail(ed|ure)?\b"),
    # Team vs. solo.
    (r"\bteam(work)?\b", r"\b(i|i'll) (prefer|work) alone\b"),
    # Positive vs. negative emotion.
    (r"\b(excited?|enthusiastic|love)\b", r"\b(boring|bored|dislike|hate)\b"),
]


def detect_contradictions(
    raw_answer: str,
    other_answers: Optional[List[Tuple[str, str]]] = None,
    # other_answers: list of (question_id, answer_text)
) -> List[Dict[str, Any]]:
    """
    Detect self-contradictions within a single answer or across answers.

    Within a single answer: checks for close proximity contradictory patterns.

    Across answers (if other_answers is provided): flags when two answers
    from the same candidate contradict each other. This requires semantic
    similarity + polarity disagreement, which is better handled by an LLM.
    Here we implement a lightweight keyword-overlap check.

    Args:
        raw_answer: The answer to check for internal contradictions.
        other_answers: Optional list of (question_id, answer_text) tuples
                       for cross-answer checks.

    Returns:
        List of contradiction records: {"type", "text1", "text2", "severity", "note"}
    """
    contradictions: List[Dict[str, Any]] = []
    text = raw_answer or ""
    lowered = text.lower()

    # Internal contradiction: search for close-proximity contradictory pairs.
    for pattern1_re, pattern2_re in _CONTRADICTION_PAIRS:
        m1 = list(re.finditer(pattern1_re, lowered))
        m2 = list(re.finditer(pattern2_re, lowered))

        for a in m1:
            for b in m2:
                distance = b.start() - a.end()
                if 0 < distance < 80:  # within 80 chars = likely internal contradiction.
                    contradictions.append(
                        {
                            "type": "self_contradiction",
                            "text1": a.group(),
                            "text2": b.group(),
                            "distance_chars": distance,
                            "severity": 0.8 if distance < 30 else 0.6,
                            "note": (
                                f"Contradictory phrases '{a.group()}' and '{b.group()}' "
                                f"found within {distance} chars."
                            ),
                        }
                    )

    # Cross-answer contradiction check (lightweight keyword overlap).
    if other_answers:
        contradictions.extend(_cross_answer_contradictions(raw_answer, other_answers))

    return contradictions


def _cross_answer_contradictions(
    anchor_answer: str,
    other_answers: List[Tuple[str, str]],
    semantic_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Lightweight cross-answer contradiction check.

    Detects:
      - Same topic mentioned with opposite polarity (e.g., salary question vs.
        experience question — both mention money with conflicting signals).
      - Conflicting availability/location answers.

    This is a heuristic stub; full semantic contradiction detection
    should be upgraded to LLM-based comparison.
    """
    contradictions: List[Dict[str, Any]] = []

    # Check for conflicting availability signals.
    availability_keywords = {
        "immediate": 1.0,
        "asap": 1.0,
        "1 month": 0.7,
        "2 weeks": 0.8,
        "notice period": -0.5,
        "3 months": -0.8,
        "6 months": -1.0,
    }

    anchor_lower = anchor_answer.lower()
    anchor_avail = [
        (kw, score) for kw, score in availability_keywords.items() if kw in anchor_lower
    ]

    for qid, other_text in other_answers:
        other_lower = other_text.lower()
        other_avail = [
            (kw, score) for kw, score in availability_keywords.items() if kw in other_lower
        ]

        if anchor_avail and other_avail:
            # Both mention availability — check for conflict.
            anchor_score = max(s for _, s in anchor_avail)
            other_score = max(s for _, s in other_avail)
            if anchor_score != other_score:
                contradictions.append(
                    {
                        "type": "cross_answer_contradiction",
                        "question_id": qid,
                        "severity": abs(anchor_score - other_score) * 0.5,
                        "note": (
                            f"Availability conflict: anchor ({anchor_answer[:40]}...) "
                            f"vs Q{qid} ({other_text[:40]}...)"
                        ),
                    }
                )

    # Note: full semantic cross-answer contradiction detection
    # requires LLM-based comparison. This stub flags keyword-level conflicts.
    if not contradictions:
        pass  # No obvious keyword conflicts found.

    return contradictions


# ---- LLM stub for future enhancement ---- #


def _llm_score_sentiment(
    raw_answer: str,
    category: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> SentimentScore:
    """
    LLM-enhanced sentiment scoring (stub).

    In production, replace with a structured JSON-mode call to an LLM:
        "Analyze the sentiment of this interview answer. Classify as
         positive/neutral/negative/mixed. Score enthusiasm, frustration,
         confidence, anxiety, calm from 0-1."
    """
    # Stub: fall back to rule-based scoring.
    return score_sentiment(raw_answer, category=category)


def _llm_detect_contradictions(
    answers: List[Tuple[str, str]],
    # (question_id, answer_text)
) -> List[Dict[str, Any]]:
    """
    LLM-based cross-answer contradiction detection (stub).

    In production, send all answers to an LLM with:
        "Identify any self-contradictions across these interview answers.
         For each contradiction, note the question IDs, the conflicting
         statements, and rate severity 0-1."
    """
    # Stub: fall back to rule-based detection.
    all_contradictions: List[Dict[str, Any]] = []
    for qid, answer in answers:
        all_contradictions.extend(detect_contradictions(answer))
    return all_contradictions
