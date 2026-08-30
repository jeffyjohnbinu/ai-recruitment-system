"""
Noisy resume handling
-----------------------
Additional noise-repair pass layered on TOP of the Day 5 TextCleaner
output (never modifies it). Targets classes of noise the Day 5 cleaner
does not attempt to fix:

  - common OCR / font-substitution character confusions (0/O, l/1/I,
    rn -> m) in tokens that are otherwise clearly a known section
    keyword or common word
  - stray repeated punctuation runs ("....", "----", "||||")
  - orphaned single-character line fragments produced by broken PDF
    text extraction (e.g. a stray "n" on its own line, as seen in the
    Day 5 sample_resume_3_noisy fixture)
  - repeated boilerplate/watermark lines that occur identically across
    many pages or many resumes in a batch (detected relative to a
    corpus, not a fixed blocklist)

Two-pass pattern: an optional dictionary-based pass (using `wordfreq` if
installed) refines whether a repaired token is a real word; without it,
the heuristic-only pass still runs, with a logged warning.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

logger = logging.getLogger("performance_optimization_engine.noisy_resume_handler")

try:
    from wordfreq import zipf_frequency  # type: ignore

    _HAS_WORDFREQ = True
except ImportError:  # pragma: no cover
    _HAS_WORDFREQ = False
    logger.warning(
        "wordfreq not installed — noisy_resume_handler falling back to "
        "heuristic-only repair without dictionary confirmation."
    )

_REPEATED_PUNCT_PATTERN = re.compile(r"([.\-_|~=*#])\1{3,}")
_ORPHAN_LINE_PATTERN = re.compile(r"^[a-zA-Z]{1}$")

# Known confusions introduced by OCR / broken glyph mapping. Applied only
# to whole tokens that become a recognizable word after substitution, so
# we never mangle genuinely correct text (e.g. a lone "0" in "iOS 0").
_OCR_SUBSTITUTIONS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\brn\b", re.IGNORECASE), "m"),
    (re.compile(r"\bl\b"), "I"),
]


@dataclass
class NoiseRepairReport:
    repeated_punct_collapsed: int = 0
    orphan_lines_removed: int = 0
    ocr_substitutions_applied: int = 0
    boilerplate_lines_removed: List[str] = field(default_factory=list)
    dictionary_backend: str = "wordfreq" if _HAS_WORDFREQ else "heuristic-only"

    def to_dict(self) -> Dict[str, object]:
        return {
            "repeated_punct_collapsed": self.repeated_punct_collapsed,
            "orphan_lines_removed": self.orphan_lines_removed,
            "ocr_substitutions_applied": self.ocr_substitutions_applied,
            "boilerplate_lines_removed": self.boilerplate_lines_removed,
            "dictionary_backend": self.dictionary_backend,
        }


def _is_real_word(token: str) -> bool:
    if not _HAS_WORDFREQ:
        # Heuristic-only fallback: accept any alphabetic token of
        # reasonable length as "plausible" rather than rejecting outright.
        return token.isalpha() and len(token) >= 2
    return zipf_frequency(token.lower(), "en") > 1.5


class NoisyResumeHandler:
    """Applies an additional repair pass to already-cleaned resume text."""

    def repair(self, cleaned_text: str) -> Tuple[str, NoiseRepairReport]:
        report = NoiseRepairReport()

        text, n = _REPEATED_PUNCT_PATTERN.subn(lambda m: m.group(1) * 3, cleaned_text)
        report.repeated_punct_collapsed = n

        lines_out: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if _ORPHAN_LINE_PATTERN.match(stripped):
                report.orphan_lines_removed += 1
                continue
            lines_out.append(line)
        text = "\n".join(lines_out)

        for pattern, replacement in _OCR_SUBSTITUTIONS:

            def _maybe_replace(match: "re.Match[str]") -> str:
                candidate = pattern.sub(replacement, match.group(0))
                if _is_real_word(candidate) and not _is_real_word(match.group(0)):
                    report.ocr_substitutions_applied += 1
                    return candidate
                return match.group(0)

            text = re.sub(r"\b\w+\b", _maybe_replace, text)

        return text, report

    def remove_corpus_boilerplate(
        self, documents: Iterable[str], min_occurrence_ratio: float = 0.6
    ) -> Tuple[List[str], NoiseRepairReport]:
        """
        Given multiple resumes' cleaned text (a batch), detect lines that
        repeat near-identically across a large fraction of the corpus —
        e.g. a template watermark or footer left in by a PDF exporter —
        and strip them, since they add noise to every downstream parser
        without carrying candidate-specific signal.
        """
        docs = list(documents)
        report = NoiseRepairReport()
        if len(docs) < 3:
            return docs, report  # not enough documents to infer boilerplate

        line_counts: Counter = Counter()
        for doc in docs:
            unique_lines = {ln.strip() for ln in doc.splitlines() if ln.strip()}
            line_counts.update(unique_lines)

        threshold = max(2, int(len(docs) * min_occurrence_ratio))
        boilerplate_lines = {
            line for line, count in line_counts.items() if count >= threshold and len(line) > 3
        }
        report.boilerplate_lines_removed = sorted(boilerplate_lines)

        cleaned_docs = []
        for doc in docs:
            kept = [ln for ln in doc.splitlines() if ln.strip() not in boilerplate_lines]
            cleaned_docs.append("\n".join(kept))

        return cleaned_docs, report
