"""
normalizer.py
-------------
Day 23 deliverable — Zecpath AI Job Portal

Implements the transcript normalization rules: raw ASR text ->
normalized text, applied per TranscriptSegment. Never deletes a
segment or drops low-confidence content -- it only cleans the text and
flags quality issues, mirroring the two-pass fallback / "flag, don't
silently drop" discipline used by resume_extraction_engine.cleaner and
performance_optimization_engine.noisy_resume_handler.

Rules implemented:
  1. Strip filler words/disfluencies (um, uh, like-as-filler, you know).
  2. Insert sentence breaks on long inter-word pause gaps, when
     word-level timestamps are supplied.
  3. Capitalize sentence starts + canonicalize known technical terms
     via an optional external alias lookup (e.g. skill_extraction_engine's
     dictionary) -- degrades to plain capitalization if none supplied.
  4. Collapse stutters/immediate word repeats ("I I think" -> "I think").
  5. Never delete low-confidence segments; only flag them (handled by
     TranscriptSegment.__post_init__ in schema.py against
     LOW_CONFIDENCE_THRESHOLD).
  6. No PII scrubbing here -- that is fairness_bias_engine's
     responsibility downstream (additive, no duplication of masking
     logic across modules).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# Standalone filler tokens removed outright. Kept short & conservative --
# "like" as a genuine word ("I would like to...") is NOT touched; only
# clearly disfluent standalone fillers are.
_FILLER_TOKENS = {"um", "umm", "uh", "uhh", "erm", "hmm"}
_FILLER_PHRASES = ["you know,", "you know", "i mean,", "kind of like"]

_TOKEN_RE = re.compile(r"\S+")
_REPEAT_WORD_RE = re.compile(r"\b(\w+)(\s+\1\b)+", re.IGNORECASE)
_SENTENCE_END_RE = re.compile(r"[.!?]\s*$")

# Long-pause threshold (seconds) between consecutive words before we
# insert a sentence break, when word-level timestamps are available.
_LONG_PAUSE_SECONDS = 0.8


class TranscriptNormalizer:
    """
    Applies the Day 23 normalization ruleset to raw ASR text.

    `term_lookup` is an optional lowercase-alias -> canonical-name dict
    (e.g. built from skill_extraction_engine.skill_dictionary.build_alias_lookup())
    used for rule 3's technical-term canonicalization. It is entirely
    optional so this module has no hard dependency on Day 9's package.
    """

    def __init__(self, term_lookup: Optional[Dict[str, str]] = None):
        self.term_lookup = term_lookup or {}

    # ------------------------------------------------------------------ #
    def normalize(
        self,
        raw_text: str,
        word_timestamps: Optional[List[Tuple[str, float, float]]] = None,
    ) -> Tuple[str, List[str]]:
        """
        Normalize one segment's raw ASR text.

        `word_timestamps`: optional list of (word, start_seconds,
        end_seconds) tuples, used only for rule 2 (pause-based sentence
        breaks). Falls back to the raw text's own spacing if omitted.

        Returns (normalized_text, notes) where notes lists which rules
        actually fired, for auditability.
        """
        notes: List[str] = []
        text = raw_text.strip()

        if not text:
            return "", notes

        # Rule 1: strip filler words/phrases
        text, filler_hits = self._strip_fillers(text)
        if filler_hits:
            notes.append(f"stripped {filler_hits} filler token(s)")

        # Rule 2: pause-based sentence breaks (only if timestamps given)
        if word_timestamps:
            text, breaks_inserted = self._insert_pause_breaks(text, word_timestamps)
            if breaks_inserted:
                notes.append(f"inserted {breaks_inserted} pause-based sentence break(s)")

        # Rule 4: collapse immediate word repeats / stutters
        text, repeats_collapsed = self._collapse_repeats(text)
        if repeats_collapsed:
            notes.append(f"collapsed {repeats_collapsed} repeated word(s)")

        # Rule 3: capitalize sentences + canonicalize known technical terms
        text = self._canonicalize_terms(text)
        text = self._capitalize_sentences(text)

        # Tidy whitespace left behind by the passes above.
        text = re.sub(r"\s{2,}", " ", text).strip()
        if text and not _SENTENCE_END_RE.search(text):
            text += "."

        return text, notes

    # ------------------------------------------------------------------ #
    # Rule 1
    # ------------------------------------------------------------------ #
    def _strip_fillers(self, text: str) -> Tuple[str, int]:
        hits = 0
        lowered_check = text.lower()
        for phrase in _FILLER_PHRASES:
            count = lowered_check.count(phrase)
            if count:
                text = re.sub(re.escape(phrase), "", text, flags=re.IGNORECASE)
                hits += count

        tokens = _TOKEN_RE.findall(text)
        kept = []
        for tok in tokens:
            bare = tok.strip(",.!?").lower()
            if bare in _FILLER_TOKENS:
                hits += 1
                continue
            kept.append(tok)
        return " ".join(kept), hits

    # ------------------------------------------------------------------ #
    # Rule 2
    # ------------------------------------------------------------------ #
    @staticmethod
    def _insert_pause_breaks(
        text: str, word_timestamps: List[Tuple[str, float, float]]
    ) -> Tuple[str, int]:
        if len(word_timestamps) < 2:
            return text, 0

        pieces: List[str] = []
        breaks = 0
        for i, (word, _start, end) in enumerate(word_timestamps):
            pieces.append(word)
            if i + 1 < len(word_timestamps):
                next_start = word_timestamps[i + 1][1]
                gap = next_start - end
                if gap >= _LONG_PAUSE_SECONDS and not word.rstrip().endswith((".", "!", "?")):
                    pieces[-1] = pieces[-1].rstrip(",") + "."
                    breaks += 1
        return " ".join(pieces), breaks

    # ------------------------------------------------------------------ #
    # Rule 3
    # ------------------------------------------------------------------ #
    def _canonicalize_terms(self, text: str) -> str:
        if not self.term_lookup:
            return text

        def _replace(match: "re.Match[str]") -> str:
            word = match.group(0)
            canonical = self.term_lookup.get(word.lower())
            return canonical if canonical else word

        return re.sub(r"[A-Za-z][A-Za-z0-9+#.]*", _replace, text)

    @staticmethod
    def _capitalize_sentences(text: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        capitalized = []
        for s in sentences:
            if s and s[0].islower():
                s = s[0].upper() + s[1:]
            capitalized.append(s)
        return " ".join(capitalized)

    # ------------------------------------------------------------------ #
    # Rule 4
    # ------------------------------------------------------------------ #
    @staticmethod
    def _collapse_repeats(text: str) -> Tuple[str, int]:
        count = 0

        def _replace(match: "re.Match[str]") -> str:
            nonlocal count
            count += 1
            return match.group(1)

        collapsed = _REPEAT_WORD_RE.sub(_replace, text)
        return collapsed, count
