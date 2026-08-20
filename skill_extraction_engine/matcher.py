"""
matcher.py
-----------
Low-level matching primitives used by the Skill Extraction Engine.

Three complementary passes over the text, in order:

1. **Phrase matching** (exact / alias) -- every canonical skill name and
   every alias in the master dictionary is compiled into a case-insensitive,
   word-boundary regex, applied longest-alias-first so multi-word phrases
   ("Machine Learning") win over any shorter word they might contain.
   Matched character spans are tracked so nothing is double-counted.

2. **Stack expansion** -- skill-stack tokens (MERN, MEAN, LAMP, ...) are
   matched the same way, then expanded into their constituent canonical
   skills, tagged "stack_inferred" so they can be scored lower than a
   direct mention.

3. **Fuzzy matching** -- any word/word-pair *not* already covered by pass 1
   is compared against the dictionary using rapidfuzz to catch spelling
   variations ("Pyth0n", "Kubernets", "Djnago") that a literal match would
   miss. A high similarity threshold keeps false positives low.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from rapidfuzz import fuzz, process

from .skill_dictionary import SKILL_STACKS, build_alias_lookup, build_stack_lookup

_WORD_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#.\-]*")

# Below this rapidfuzz similarity ratio, a fuzzy candidate is discarded as
# noise rather than treated as a spelling variant of a real skill.
FUZZY_MATCH_THRESHOLD = 82.0

# Fuzzy matching is restricted to tokens in this length band. Very short
# tokens produce too many accidental matches ("R" vs "Rs"), very long ones
# are already well covered by exact phrase matching.
_FUZZY_MIN_LEN = 4
_FUZZY_MAX_LEN = 24

# Common English words that sit uncomfortably close (edit-distance-wise) to
# a real skill name -- "unit" vs "JUnit", "test" vs "pytest" -- but almost
# always appear as ordinary language rather than a spelling mistake. Fuzzy
# matching skips these outright rather than relying on the ratio threshold
# alone to filter them out.
_FUZZY_STOPWORDS = {
    "unit",
    "test",
    "tests",
    "testing",
    "class",
    "code",
    "team",
    "role",
    "tool",
    "tools",
    "plan",
    "model",
    "models",
    "data",
    "app",
    "apps",
    "use",
    "used",
    "using",
    "user",
    "users",
    "task",
    "tasks",
    "part",
    "type",
    "types",
    "case",
    "cases",
    "line",
    "lines",
    "level",
    "story",
}


@dataclass
class SkillMention:
    canonical: str
    match_type: str  # "exact" | "alias" | "fuzzy" | "stack_inferred"
    matched_text: str
    start: int
    end: int
    fuzzy_ratio: float | None = None
    stack_source: str | None = None


class SkillMatcher:
    """Compiles the dictionary once, then matches resume text against it."""

    def __init__(self) -> None:
        self._alias_lookup: Dict[str, str] = build_alias_lookup()
        self._stack_lookup: Dict[str, str] = build_stack_lookup()

        self._phrase_pattern = self._compile_phrase_pattern(self._alias_lookup.keys())
        self._stack_pattern = self._compile_phrase_pattern(self._stack_lookup.keys())

        # Flat list of (alias_text, canonical_skill) used as the fuzzy
        # candidate pool -- restricted to single tokens / short phrases so
        # ratio comparisons stay meaningful.
        self._fuzzy_candidates: List[str] = [
            alias for alias in self._alias_lookup if _FUZZY_MIN_LEN <= len(alias) <= _FUZZY_MAX_LEN
        ]

    @staticmethod
    def _compile_phrase_pattern(phrases) -> re.Pattern:
        # Longest-first so multi-word aliases are preferred over any
        # shorter alias that happens to be a substring/prefix.
        ordered = sorted(set(phrases), key=len, reverse=True)
        escaped = [re.escape(p) for p in ordered if p]
        if not escaped:
            # Pattern that never matches anything.
            return re.compile(r"(?!x)x")
        pattern = r"(?<![A-Za-z0-9_])(" + "|".join(escaped) + r")(?![A-Za-z0-9_])"
        return re.compile(pattern, re.IGNORECASE)

    # ------------------------------------------------------------------ #
    def match_phrases(self, text: str) -> Tuple[List[SkillMention], Set[Tuple[int, int]]]:
        """Exact/alias phrase matches. Returns mentions + the character
        spans they cover (so later passes can avoid re-matching them)."""
        mentions: List[SkillMention] = []
        covered: Set[Tuple[int, int]] = set()

        for m in self._phrase_pattern.finditer(text):
            matched_text = m.group(0)
            canonical = self._alias_lookup[matched_text.lower()]
            match_type = "exact" if matched_text.lower() == canonical.lower() else "alias"
            mentions.append(
                SkillMention(
                    canonical=canonical,
                    match_type=match_type,
                    matched_text=matched_text,
                    start=m.start(),
                    end=m.end(),
                )
            )
            covered.add((m.start(), m.end()))

        return mentions, covered

    def match_stacks(self, text: str) -> List[SkillMention]:
        """Stack tokens (MERN, MEAN, ...), expanded into constituent skills."""
        mentions: List[SkillMention] = []
        for m in self._stack_pattern.finditer(text):
            matched_text = m.group(0)
            stack_name = self._stack_lookup[matched_text.lower()]
            for skill in SKILL_STACKS[stack_name]:
                mentions.append(
                    SkillMention(
                        canonical=skill,
                        match_type="stack_inferred",
                        matched_text=matched_text,
                        start=m.start(),
                        end=m.end(),
                        stack_source=stack_name,
                    )
                )
        return mentions

    def match_fuzzy(self, text: str, covered_spans: Set[Tuple[int, int]]) -> List[SkillMention]:
        """
        Fuzzy spelling-variant matching over tokens not already covered by
        an exact/alias match. Single tokens and adjacent-word bigrams are
        both tried, since some aliases ("node js") are two words.
        """
        mentions: List[SkillMention] = []
        tokens = list(_WORD_TOKEN_RE.finditer(text))

        candidates = [(t.group(0), t.start(), t.end()) for t in tokens]
        # Adjacent-word bigrams (e.g. "node js", "power bi")
        for i in range(len(tokens) - 1):
            a, b = tokens[i], tokens[i + 1]
            if b.start() - a.end() <= 1:  # separated by at most one space
                candidates.append((f"{a.group(0)} {b.group(0)}", a.start(), b.end()))

        for word, start, end in candidates:
            span = (start, end)
            if any(not (end <= cs or start >= ce) for cs, ce in covered_spans):
                continue  # overlaps an already-matched exact/alias span
            if not (_FUZZY_MIN_LEN <= len(word) <= _FUZZY_MAX_LEN):
                continue
            # Skip tokens that are themselves an exact dictionary alias --
            # those are handled (or intentionally excluded) by pass 1.
            if word.lower() in self._alias_lookup:
                continue
            if word.lower() in _FUZZY_STOPWORDS:
                continue

            result = process.extractOne(word.lower(), self._fuzzy_candidates, scorer=fuzz.ratio)
            if result is None:
                continue
            matched_alias, ratio, _ = result
            if ratio >= FUZZY_MATCH_THRESHOLD:
                canonical = self._alias_lookup[matched_alias]
                mentions.append(
                    SkillMention(
                        canonical=canonical,
                        match_type="fuzzy",
                        matched_text=word,
                        start=start,
                        end=end,
                        fuzzy_ratio=ratio,
                    )
                )
                covered_spans.add(span)

        return mentions
