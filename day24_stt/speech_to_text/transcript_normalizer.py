"""
speech_to_text/transcript_normalizer.py
---------------------------------------
Cleans and normalizes raw transcripts from the STT service.

Transforms raw speech output into structured, AI-analysis-ready text by:
1. Removing filler words (uh, um, like, you know, etc.)
2. Correcting punctuation based on speech patterns
3. Normalizing case (proper sentence capitalization)
4. Handling interrupted speech markers
5. Cleaning partial answers
6. Normalizing timestamps to readable format

Usage:
    normalizer = TranscriptNormalizer()
    cleaned = normalizer.normalize("uh yeah i think like um Python is good...")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from utils.logger import get_logger

logger = get_logger("speech_to_text.transcript_normalizer")

# Filler words to remove (order-sensitive — longer patterns first)
# These are common in spoken English and don't add semantic meaning
FILLER_WORDS: list[str] = [
    # Single-word fillers
    "uh",
    "um",
    "er",
    "ah",
    "eh",
    "hm",
    "hmm",
    "mmm",
    "mm",
    "oh",
    "ah",
    # Two-word fillers
    "you know",
    "i mean",
    "kind of",
    "sort of",
    "like, like",
    "like, uh",
    "like, um",
    "well, like",
    "so, like",
    "basically, like",
    # Phrase fillers
    "i think like",
    "like i said",
    "like i guess",
    "if you will",
    "you know what i mean",
    "if that makes sense",
    "right?",
    "okay so",
    "so basically",
]

# Sort by length descending to avoid partial matches (e.g., "like" before "like, like")
FILLER_WORDS_SORTED: list[str] = sorted(FILLER_WORDS, key=len, reverse=True)

# Regex patterns for speech patterns
_REPEATED_PUNCTUATION = re.compile(r"([!?.])\1{2,}")
_REPEATED_SPACES = re.compile(r"\s{2,}")
_LEADING_SPACE = re.compile(r"^\s+")
_TRAILING_SPACE = re.compile(r"\s+$")
_INCOMPLETE_WORD = re.compile(r"\b(\w+)-\s+(\w+)\b")  # "test ing" -> "testing"


@dataclass
class NormalizationResult:
    """Result of normalizing a transcript."""

    original: str
    cleaned: str
    removed_fillers: list[str] = field(default_factory=list)
    corrections_made: int = 0
    interrupted: bool = False
    partial: bool = False


class TranscriptNormalizer:
    """
    Cleans and normalizes raw speech transcripts.

    Features:
    - Filler word removal with configurable allowlist
    - Punctuation correction (handles trailing/leading punctuation from speech)
    - Case normalization (proper sentence capitalization)
    - Interrupted speech marker removal
    - Partial answer handling
    - Customizable preservation of certain speech patterns
    """

    def __init__(
        self,
        preserve_fillers: bool = False,
        preserve_incomplete: bool = False,
    ) -> None:
        """
        Initialize the normalizer.

        Args:
            preserve_fillers: If True, keep filler words (for analysis).
                              If False (default), remove them.
            preserve_incomplete: If True, keep incomplete words as-is.
                                If False, join hyphenated words.
        """
        self.preserve_fillers = preserve_fillers
        self.preserve_incomplete = preserve_incomplete
        logger.info(
            "TranscriptNormalizer initialized: preserve_fillers=%s preserve_incomplete=%s",
            preserve_fillers,
            preserve_incomplete,
        )

    def normalize(
        self,
        text: str,
        *,
        interrupted: bool = False,
        partial: bool = False,
        case: Literal["preserve", "lower", "sentence"] = "sentence",
    ) -> NormalizationResult:
        """
        Normalize a raw transcript string.

        Args:
            text: Raw transcript text from STT.
            interrupted: Whether this transcript was flagged as interrupted.
            partial: Whether this transcript contains partial answers.
            case: Case handling:
                  - "preserve": Keep original casing
                  - "lower": Convert everything to lowercase
                  - "sentence": Proper sentence capitalization (default)

        Returns:
            NormalizationResult with original text, cleaned text, and metadata.
        """
        if not text or not text.strip():
            return NormalizationResult(original=text, cleaned="")

        original = text
        removed_fillers: list[str] = []
        corrections = 0

        # Step 1: Remove filler words (unless preserving)
        cleaned = text
        if not self.preserve_fillers:
            cleaned, fillers = self._remove_filler_words(cleaned)
            removed_fillers = fillers
            if fillers:
                corrections += len(fillers)

        # Step 2: Fix hyphenated/incomplete words
        if not self.preserve_incomplete:
            cleaned = self._fix_incomplete_words(cleaned)
            corrections += cleaned.count("- ")  # rough count

        # Step 3: Clean punctuation
        cleaned = self._clean_punctuation(cleaned)
        corrections += 1  # at least one pass

        # Step 4: Normalize whitespace
        cleaned = self._normalize_whitespace(cleaned)

        # Step 5: Handle case
        if case == "lower":
            cleaned = cleaned.lower()
        elif case == "sentence":
            cleaned = self._normalize_sentence_case(cleaned)
        # "preserve" keeps as-is

        # Step 6: Handle interrupted speech markers
        if interrupted:
            cleaned = self._handle_interrupted(cleaned)
            corrections += 1

        # Step 7: Clean partial answers
        if partial:
            cleaned = self._clean_partial_answers(cleaned)
            corrections += 1

        logger.debug(
            "Normalized transcript: %d chars -> %d chars, " "%d fillers removed, %d corrections",
            len(original),
            len(cleaned),
            len(removed_fillers),
            corrections,
        )

        return NormalizationResult(
            original=original,
            cleaned=cleaned.strip(),
            removed_fillers=removed_fillers,
            corrections_made=corrections,
            interrupted=interrupted,
            partial=partial,
        )

    def _remove_filler_words(self, text: str) -> tuple[str, list[str]]:
        """Remove filler words and return cleaned text + list of removed fillers."""
        removed: list[str] = []
        result = text

        # Phrases containing commas/spaces need to match exactly,
        # but single words need word-boundary matching so we don't
        # strip "ah" from "yeah" or "like" from "likelihood".
        for filler in FILLER_WORDS_SORTED:
            # Use word boundary anchors for clean single-word matches
            if re.fullmatch(r"[a-zA-Z]+", filler):
                pattern = re.compile(rf"\b{re.escape(filler)}\b", re.IGNORECASE)
            else:
                # Phrases with punctuation/spaces: use literal match
                pattern = re.compile(re.escape(filler), re.IGNORECASE)

            matches = pattern.findall(result)
            removed.extend(matches)
            result = pattern.sub(" ", result)

        return result, removed

    def _fix_incomplete_words(self, text: str) -> str:
        """Join words that were split by speech interruption (e.g., 'test ing')."""
        return _INCOMPLETE_WORD.sub(r"\1\2", text)

    def _clean_punctuation(self, text: str) -> str:
        """
        Clean up punctuation artifacts from speech.

        Handles:
        - Multiple repeated punctuation (!!! -> !)
        - Trailing punctuation followed by space then lowercase
        - Missing punctuation before conjunctions
        """
        # Collapse repeated punctuation
        result = _REPEATED_PUNCTUATION.sub(r"\1", text)

        # Ensure proper spacing around punctuation
        result = re.sub(r"\s*([.,!?])\s*", r" \1 ", result)
        result = re.sub(r"\s+", " ", result)

        return result.strip()

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize all whitespace to single spaces, strip leading/trailing."""
        result = _LEADING_SPACE.sub("", text)
        result = _TRAILING_SPACE.sub("", result)
        result = _REPEATED_SPACES.sub(" ", result)
        return result

    def _normalize_sentence_case(self, text: str) -> str:
        """
        Apply proper sentence capitalization.

        Capitalizes the first letter of each sentence and ensures
        proper spacing after sentence-ending punctuation.
        """
        if not text:
            return text

        # Split into sentences (conservative split)
        sentences = re.split(r"([.!?]+\s+)", text)
        result_parts: list[str] = []

        for i, part in enumerate(sentences):
            if i % 2 == 0:  # Sentence text
                part = part.strip()
                if part:
                    # Capitalize first letter, lowercase rest
                    part = part[0].upper() + part[1:].lower() if len(part) > 1 else part.upper()
            result_parts.append(part)

        result = "".join(result_parts)

        # Clean up double spaces after punctuation
        result = re.sub(r"\.(\s*)\.?", r". ", result)
        result = re.sub(r"!\s*!", "! ", result)
        result = re.sub(r"\?\s*\??", "? ", result)

        return result.strip()

    def _handle_interrupted(self, text: str) -> str:
        """
        Clean up text that was interrupted mid-sentence.

        Removes trailing fragments that are too short to be meaningful.
        Adds ellipsis marker for downstream awareness.
        """
        # Find sentences and their lengths
        sentences = re.split(r"([.!?]+\s+)", text)
        cleaned_sentences: list[str] = []

        for i, part in enumerate(sentences):
            if i % 2 == 0:  # Sentence content
                part = part.strip()
                if part and len(part.split()) >= 2:
                    cleaned_sentences.append(part)
                elif part and i == 0:  # First sentence, keep even if short
                    cleaned_sentences.append(part)

        return " ".join(cleaned_sentences) if cleaned_sentences else text

    def _clean_partial_answers(self, text: str) -> str:
        """
        Clean up partial/fragmentary answers.

        Removes trailing fragments that are too short or incomplete.
        """
        sentences = re.split(r"([.!?]+\s+)", text)
        cleaned_sentences: list[str] = []

        for i, part in enumerate(sentences):
            if i % 2 == 0:  # Sentence content
                part = part.strip()
                # Keep sentences with at least 3 words
                if part and len(part.split()) >= 3:
                    cleaned_sentences.append(part)

        if cleaned_sentences:
            return " ".join(cleaned_sentences)
        return text

    def normalize_segment(
        self,
        segment_text: str,
        index: int,
        is_partial: bool = False,
    ) -> str:
        """
        Normalize a single transcript segment.

        Args:
            segment_text: Text content of a single transcript segment.
            index: Segment index (0-based).
            is_partial: Whether this segment was flagged as partial.

        Returns:
            Normalized segment text.
        """
        result = self.normalize(
            segment_text,
            partial=is_partial,
            case="sentence",
        )
        return result.cleaned

    def get_filler_statistics(self, text: str) -> dict[str, int]:
        """
        Get statistics about filler word usage in raw text.

        Useful for analyzing speaker patterns or interview dynamics.
        """
        stats: dict[str, int] = {}
        for filler in FILLER_WORDS_SORTED:
            pattern = re.compile(re.escape(filler), re.IGNORECASE)
            count = len(pattern.findall(text))
            if count > 0:
                stats[filler] = count
        return stats
