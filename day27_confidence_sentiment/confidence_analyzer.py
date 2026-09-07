"""
confidence_analyzer.py
---------------------
Day 27 deliverable — Zecpath AI Job Portal

Rule-based confidence analysis components:

1. Hesitation detection  — filler words, pauses, repetitions, repairs, false starts
2. Pace measurement      — words-per-second, ideal-pace scoring, length buckets
3. Uncertainty detection — hedges, doubt markers, vague quantifiers, self-contradictions

All functions are pure, deterministic, and testable. No I/O, no LLM calls.
LLM stubs are prefixed with `_llm_` for future enhancement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .formats import HesitationPattern, PaceMetrics, UncertaintySignal

# ---- shared regex ---- #

# Filler words (spoken hesitation markers).
_FILLER_RE = re.compile(
    r"\b(um+|uh+|er+|ah+|hmm+|hm+|uhh+|erm+)\b",
    re.IGNORECASE,
)

# Pauses and mid-sentence breaks (parenthetical or stuttered).
_PAUSE_RE = re.compile(
    r"\b(like|you know|i mean)\b",
    re.IGNORECASE,
)

# Repetitions: same word (or 2-word phrase) appearing twice in quick succession.
# Matches patterns like "I I", "the the", "I think I think".
_REPETITION_RE = re.compile(
    r"\b(\w+)(\s+\1){1,}\b",
    re.IGNORECASE,
)

# Stutters: single letter repeated before a word: "w-w-working", "b-being".
_STUTTER_RE = re.compile(
    r"\b([a-zA-Z])-\1{1,}([a-zA-Z]+)\b",
)

# Repairs / self-corrections: "I mean", "or rather", "actually", "sorry", "let me rephrase".
_REPAIR_RE = re.compile(
    r"\b("
    r"i mean|or rather|actually|sorry|let me|let me rephrase|"
    r"let me clarify|what i meant|what i mean|correct me|"
    r"rather i|no i|wait|i'm sorry|excuse me|"
    r"i should say|to be honest|in other words"
    r")\b",
    re.IGNORECASE,
)

# False starts: truncated beginning of a sentence, e.g. "I was — I was working".
_FALSE_START_RE = re.compile(
    r"(?<!\w)—\s*|\b(\w+)\s+—",
)

# Hedging words (uncertainty markers).
_HEDGE_RE = re.compile(
    r"\b("
    r"maybe|perhaps|possibly|might be|might have|could be|could have|"
    r"would be|would have|should be|should have|"
    r"kind of|sort of|type of|kind of a|sort of a|"
    r"i think|i guess|i suppose|i believe|i feel|"
    r"it seems like|it seems that|it appears|it appears to be|"
    r"roughly|approximately|around|about|"
    r"not sure|not certain|not really|not particularly|"
    r"i'm not sure|i'm not certain|i'm not really|"
    r"not sure if|not sure whether|"
    r"it depends|who knows|"
    r"allegedly|supposedly|theoretically|vaguely"
    r")\b",
    re.IGNORECASE,
)

# Doubt / hesitation verbal phrases.
_DOUBT_RE = re.compile(
    r"\b("
    r"i don't know|i'm not sure|i'm not certain|"
    r"i can't remember|i cannot recall|i'm not sure i can|"
    r"i'm not sure if|i'm not sure whether|"
    r"i don't remember|i don't recall|"
    r"let me think|give me a second|give me a moment|"
    r"that's a good question|that's tough|that's hard|that's difficult|"
    r"i'm not 100% sure|i'm not entirely sure|"
    r"to be honest|honestly speaking|if i'm being honest|"
    r"i'm not sure to be honest|"
    r"i'm kind of|i'm sort of"
    r")\b",
    re.IGNORECASE,
)

# Vague quantifiers / weak assertions.
_VAGUE_QUANT_RE = re.compile(
    r"\b("
    r"a few|several|some|most|a couple|a bit|a little|"
    r"a lot|many|much|very|really|pretty|quite|rather|fairly|"
    r"basically|mostly|generally|typically|usually|often|sometimes|"
    r"occasionally|rarely|now and then|from time to time|once in a while"
    r")\b",
    re.IGNORECASE,
)

# Negation patterns for contradiction detection.
_NEGATION_RE = re.compile(
    r"\b(not|never|no|none|nothing|nowhere|neither|nobody|hardly|barely|rarely)\b",
    re.IGNORECASE,
)


# ---- severity helpers ---- #


def _hedge_severity(text: str) -> float:
    """Return a severity 0-1 for hedging intensity."""
    heavy = {
        "i don't know",
        "who knows",
        "i'm not sure",
        "i'm not certain",
        "i can't remember",
        "i cannot recall",
        "it depends",
    }
    medium = {
        "maybe",
        "perhaps",
        "might be",
        "could be",
        "would be",
        "i think",
        "i guess",
        "i believe",
        "i feel",
        "not sure",
        "not certain",
        "not really",
        "let me think",
        "give me a moment",
    }
    mild = {
        "kind of",
        "sort of",
        "roughly",
        "approximately",
        "around",
        "about",
        "basically",
        "pretty",
        "quite",
    }
    t = text.strip().lower()
    if t in heavy:
        return 0.85
    if t in medium:
        return 0.55
    if t in mild:
        return 0.30
    return 0.40  # fallback for unknown hedges


# ---- 1. Hesitation detection ---- #


@dataclass
class HesitationDetail:
    """Internal hesitation detection result."""

    pattern_type: str
    text: str
    position: int
    severity: float
    note: str = ""


def detect_hesitation_patterns(
    raw_answer: str,
    *,
    include_stutters: bool = True,
) -> List[HesitationPattern]:
    """
    Detect hesitation patterns in a candidate's answer.

    Pattern types detected:
      - filler   : um, uh, er, ah, hmm
      - pause    : like, you know, i mean (verbal pauses)
      - repetition : word/phrase repeated twice
      - repair   : i mean, or rather, actually, sorry
      - false_start : mid-sentence dash or truncated word

    Args:
        raw_answer: The candidate's raw text answer.
        include_stutters: Whether to detect letter-stutters (w-w-working).

    Returns:
        List of HesitationPattern records sorted by position.
    """
    patterns: List[HesitationDetail] = []
    text = raw_answer or ""

    # 1a. Filler words.
    for m in _FILLER_RE.finditer(text):
        severity = min(1.0, 0.2 + 0.1 * (len(m.group()) - 1))
        patterns.append(
            HesitationDetail(
                pattern_type="filler",
                text=m.group(),
                position=m.start(),
                severity=severity,
                note="Spoken filler word.",
            )
        )

    # 1b. Verbal pauses.
    for m in _PAUSE_RE.finditer(text):
        patterns.append(
            HesitationDetail(
                pattern_type="pause",
                text=m.group(),
                position=m.start(),
                severity=0.35,
                note="Verbal pause / filler phrase.",
            )
        )

    # 1c. Repetitions (same word repeated).
    for m in _REPETITION_RE.finditer(text):
        word = m.group().strip()
        # Ignore short common words that appear together normally.
        skip = {"a", "an", "the", "is", "was", "i", "to", "of", "and", "or"}
        if word.lower() not in skip:
            patterns.append(
                HesitationDetail(
                    pattern_type="repetition",
                    text=m.group(),
                    position=m.start(),
                    severity=0.5,
                    note=f"Repeated word/phrase: '{word}'.",
                )
            )

    # 1d. Stutters (letter repeated: w-w-working).
    if include_stutters:
        for m in _STUTTER_RE.finditer(text):
            patterns.append(
                HesitationDetail(
                    pattern_type="repetition",
                    text=m.group(),
                    position=m.start(),
                    severity=0.6,
                    note=f"Stutter: letter repeated in '{m.group()}'.",
                )
            )

    # 1e. Repairs / self-corrections.
    for m in _REPAIR_RE.finditer(text):
        t = m.group().strip().lower()
        note_map = {
            "i mean": "Self-correction: 'I mean'.",
            "or rather": "Self-correction: 'or rather'.",
            "actually": "Self-correction: 'actually'.",
            "sorry": "Apology / repair marker.",
            "let me": "Self-repair: seeking clarification.",
            "let me rephrase": "Rephrasing.",
            "let me clarify": "Clarification sought.",
            "what i meant": "Retraction / clarification.",
            "what i mean": "Rephrasing.",
            "correct me": "Uncertainty signalled.",
            "i'm sorry": "Apology / hesitation.",
            "excuse me": "Politeness / hesitation.",
        }
        patterns.append(
            HesitationDetail(
                pattern_type="repair",
                text=m.group(),
                position=m.start(),
                severity=0.55,
                note=note_map.get(t, "Self-correction / repair marker."),
            )
        )

    # 1f. False starts (dash).
    for m in _FALSE_START_RE.finditer(text):
        patterns.append(
            HesitationDetail(
                pattern_type="false_start",
                text=m.group(),
                position=m.start(),
                severity=0.4,
                note="False start or mid-sentence dash.",
            )
        )

    # Sort by position for consistent output.
    patterns.sort(key=lambda p: p.position)

    # Deduplicate overlapping matches (prefer higher severity).
    filtered: List[HesitationPattern] = []
    last_end = -1
    for p in patterns:
        if p.position >= last_end:
            filtered.append(
                HesitationPattern(
                    pattern_type=p.pattern_type,
                    text=p.text,
                    position=p.position,
                    severity=p.severity,
                    note=p.note,
                )
            )
            last_end = p.position + len(p.text)

    return filtered


# ---- 2. Pace measurement ---- #

# Ideal speaking pace: 120-150 words/min = 2.0-2.5 wps.
_IDEAL_WPS_MIN = 1.5
_IDEAL_WPS_MAX = 3.0

# Ideal sentence length: 10-20 words per sentence for spoken answers.
_IDEAL_AVG_SENT_MIN = 8.0
_IDEAL_AVG_SENT_MAX = 22.0

# Word count buckets.
_MIN_WORDS_SHORT = 3
_MAX_WORDS_IDEAL = 60
_MAX_WORDS_LONG = 120

# Sentence-splitting: split on . ! ? while preserving abbreviations.
_SENTENCE_SPLIT_RE = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+")


def measure_pace(
    raw_answer: str,
    duration_seconds: Optional[float] = None,
    expected_min_words: int = 3,
    expected_max_words: int = 60,
) -> PaceMetrics:
    """
    Measure response length and speaking pace.

    Args:
        raw_answer: The candidate's answer text.
        duration_seconds: Optional audio duration for wps calculation.
        expected_min_words: Minimum expected words for a normal answer.
        expected_max_words: Upper threshold for a normal-length answer.

    Returns:
        PaceMetrics with all pace/length scores.
    """
    text = (raw_answer or "").strip()

    # Word count.
    word_re = re.compile(r"[A-Za-z][A-Za-z0-9+#.]{0,}")
    words = word_re.findall(text)
    word_count = len(words)
    char_count = len(text.replace(" ", ""))

    # Sentence count.
    sentences = _SENTENCE_SPLIT_RE.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)

    # Avg sentence length.
    avg_sent_len = word_count / max(sentence_count, 1)

    # Words per second.
    wps: Optional[float] = None
    if duration_seconds and duration_seconds > 0:
        wps = word_count / duration_seconds

    # Pace label + score.
    pace_label: str
    pace_score: float

    if wps is not None:
        if wps < _IDEAL_WPS_MIN * 0.5:
            pace_label, pace_score = "too_slow", 0.2
        elif wps < _IDEAL_WPS_MIN:
            pace_label, pace_score = "slow", 0.6
        elif wps > _IDEAL_WPS_MAX * 1.5:
            pace_label, pace_score = "too_fast", 0.2
        elif wps > _IDEAL_WPS_MAX:
            pace_label, pace_score = "fast", 0.65
        else:
            pace_label, pace_score = "normal", 1.0
    else:
        # No duration: infer from avg sentence length.
        if avg_sent_len < _IDEAL_AVG_SENT_MIN * 0.5:
            pace_label, pace_score = "slow", 0.6
        elif avg_sent_len > _IDEAL_AVG_SENT_MAX * 1.5:
            pace_label, pace_score = "fast", 0.65
        else:
            pace_label, pace_score = "normal", 0.8

    # Length label + score.
    length_label: str
    length_score: float

    if word_count < _MIN_WORDS_SHORT:
        length_label, length_score = "too_short", 0.1
    elif word_count < expected_min_words:
        length_label, length_score = "short", 0.5
    elif word_count > expected_max_words * 2:
        length_label, length_score = "too_long", 0.2
    elif word_count > expected_max_words:
        length_label, length_score = "long", 0.65
    else:
        length_label, length_score = "normal", 1.0

    # Build note.
    note_parts = []
    if wps is not None:
        if pace_label in ("too_slow", "slow"):
            note_parts.append(
                f"Candidate spoke at {wps:.1f} wps (ideal: {_IDEAL_WPS_MIN}-{_IDEAL_WPS_MAX})."
            )
        elif pace_label == "too_fast":
            note_parts.append(f"Candidate spoke at {wps:.1f} wps (too fast).")
    if length_label in ("too_short", "short"):
        note_parts.append(f"Answer is {length_label} ({word_count} words).")
    elif length_label in ("too_long", "long"):
        note_parts.append(f"Answer is {length_label} ({word_count} words).")

    return PaceMetrics(
        word_count=word_count,
        char_count=char_count,
        sentence_count=sentence_count,
        avg_sentence_length=round(avg_sent_len, 2),
        duration_seconds=duration_seconds,
        words_per_second=wps,
        pace_label=pace_label,
        pace_score=pace_score,
        length_label=length_label,
        length_score=length_score,
        note=" ".join(note_parts),
    )


# ---- 3. Uncertainty detection ---- #


def detect_uncertainty(
    raw_answer: str,
    category: Optional[str] = None,
) -> List[UncertaintySignal]:
    """
    Detect uncertainty markers: hedges, doubt phrases, vague quantifiers.

    Args:
        raw_answer: The candidate's answer text.
        category: Optional category hint (experience, salary, etc.) for targeted checks.

    Returns:
        List of UncertaintySignal records sorted by position.
    """
    signals: List[UncertaintySignal] = []
    text = raw_answer or ""

    # 3a. Hedging.
    for m in _HEDGE_RE.finditer(text):
        t = m.group().strip()
        severity = _hedge_severity(t)
        note = f"Hedge detected: '{t}'."
        signals.append(
            UncertaintySignal(
                signal_type="hedge",
                text=m.group(),
                position=m.start(),
                severity=severity,
                note=note,
            )
        )

    # 3b. Doubt / verbal uncertainty.
    for m in _DOUBT_RE.finditer(text):
        signals.append(
            UncertaintySignal(
                signal_type="doubt",
                text=m.group(),
                position=m.start(),
                severity=0.75,
                note="Doubt / uncertainty phrase.",
            )
        )

    # 3c. Vague quantifiers (only flag if overused).
    vague_hits = list(_VAGUE_QUANT_RE.finditer(text))
    if len(vague_hits) >= 3:
        for m in vague_hits:
            signals.append(
                UncertaintySignal(
                    signal_type="vague_quantifier",
                    text=m.group(),
                    position=m.start(),
                    severity=0.4,
                    note="Vague quantifier (one of 3+ detected).",
                )
            )
    elif vague_hits:
        for m in vague_hits:
            signals.append(
                UncertaintySignal(
                    signal_type="vague_quantifier",
                    text=m.group(),
                    position=m.start(),
                    severity=0.2,
                    note="Vague quantifier (isolated).",
                )
            )

    # Sort by position.
    signals.sort(key=lambda s: s.position)

    # Deduplicate overlapping.
    filtered: List[UncertaintySignal] = []
    last_end = -1
    for s in signals:
        if s.position >= last_end:
            filtered.append(s)
            last_end = s.position + len(s.text)

    return filtered


# ---- 4. High-level analyze_confidence ---- #


def analyze_confidence(
    raw_answer: str,
    duration_seconds: Optional[float] = None,
    category: Optional[str] = None,
    expected_min_words: int = 3,
    expected_max_words: int = 60,
) -> Tuple[List[HesitationPattern], PaceMetrics, List[UncertaintySignal]]:
    """
    Convenience wrapper: run all three analysis steps at once.

    Returns (hesitation_patterns, pace_metrics, uncertainty_signals).
    """
    hesitations = detect_hesitation_patterns(raw_answer)
    pace = measure_pace(
        raw_answer,
        duration_seconds=duration_seconds,
        expected_min_words=expected_min_words,
        expected_max_words=expected_max_words,
    )
    uncertainties = detect_uncertainty(raw_answer, category=category)
    return hesitations, pace, uncertainties
