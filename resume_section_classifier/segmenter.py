"""
Section segmentation
----------------------
Splits cleaned resume text into a sequence of `RawBlock`s, each tagged
with a canonical section label, using a two-pass strategy:

Pass 1 (rule-based): walk the text line by line. Any line that matches
a known section heading (see `rules.match_heading`) opens a new block
under that label. This handles the common case — a resume with clear,
even if inconsistently-worded, section headings — cheaply and with
100% confidence.

Pass 2 (NLP-based fallback): any stretch of text that pass 1 could not
attach to a heading (no heading found yet -> "Header" zone, or a
resume with no headings at all) is split into paragraphs on blank
lines and each paragraph is scored independently by
`nlp_classifier.classify_block`. Consecutive paragraphs that land on
the same label are merged into a single block.

Every block also carries the NLP classifier's *independent* opinion
(`nlp_label`/`nlp_confidence`) even when the block came from a rule
match, so callers can flag heading/content disagreement (e.g. a
"Skills" heading sitting over content that reads like Experience —
a common copy-paste mistake in real resumes) instead of trusting the
heading blindly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from . import nlp_classifier, rules


@dataclass
class RawBlock:
    label: str
    text: str
    method: str  # "rule_heading" | "nlp" | "header_heuristic"
    confidence: float
    start_line: int
    end_line: int
    nlp_label: Optional[str] = None
    nlp_confidence: Optional[float] = None
    heading_text: Optional[str] = None
    flags: List[str] = field(default_factory=list)


def _split_paragraphs(lines: List[str], start_line: int) -> List[tuple[int, int, str]]:
    """Split a run of lines into (start, end, text) paragraphs on blank lines."""
    paragraphs: List[tuple[int, int, str]] = []
    buf: List[str] = []
    buf_start = start_line
    for i, line in enumerate(lines):
        idx = start_line + i
        if line.strip():
            if not buf:
                buf_start = idx
            buf.append(line)
        else:
            if buf:
                paragraphs.append((buf_start, idx - 1, "\n".join(buf)))
                buf = []
    if buf:
        paragraphs.append((buf_start, start_line + len(lines) - 1, "\n".join(buf)))
    return paragraphs


def _classify_unlabeled_span(lines: List[str], start_line: int, is_leading: bool) -> List[RawBlock]:
    """
    Run the NLP fallback over a span of text with no rule-matched
    heading. `is_leading` marks the very top of the resume, where a
    contact-info heuristic takes priority over the general lexicon.
    """
    blocks: List[RawBlock] = []
    for p_start, p_end, para_text in _split_paragraphs(lines, start_line):
        if not para_text.strip():
            continue

        if is_leading and p_start == start_line and rules.looks_like_contact_line(para_text):
            blocks.append(
                RawBlock(
                    label="Header",
                    text=para_text,
                    method="header_heuristic",
                    confidence=0.9,
                    start_line=p_start,
                    end_line=p_end,
                    nlp_label="Header",
                    nlp_confidence=0.9,
                )
            )
            continue

        result = nlp_classifier.classify_block(para_text)
        label = result.label
        # A leading, short, heading-less paragraph with no strong
        # signal is very likely the header/summary blurb rather than
        # genuinely uncategorized content.
        if is_leading and label == "Uncategorized" and len(para_text.split()) < 60:
            label = "Summary" if len(para_text.split()) > 8 else "Header"

        blocks.append(
            RawBlock(
                label=label,
                text=para_text,
                method="nlp",
                confidence=result.confidence,
                start_line=p_start,
                end_line=p_end,
                nlp_label=result.label,
                nlp_confidence=result.confidence,
            )
        )

    return _merge_adjacent(blocks)


def _merge_adjacent(blocks: List[RawBlock]) -> List[RawBlock]:
    """Merge consecutive NLP blocks that share the same label."""
    if not blocks:
        return blocks
    merged = [blocks[0]]
    for b in blocks[1:]:
        last = merged[-1]
        if b.label == last.label and b.method == last.method == "nlp":
            last.text = last.text + "\n\n" + b.text
            last.end_line = b.end_line
            last.confidence = round((last.confidence + b.confidence) / 2, 3)
        else:
            merged.append(b)
    return merged


class SectionSegmenter:
    """Segments cleaned resume text into labeled `RawBlock`s."""

    def segment(self, text: str) -> List[RawBlock]:
        lines = text.splitlines()
        n = len(lines)

        # 1. Locate every rule-matched heading line.
        heading_positions: List[tuple[int, str, str]] = (
            []
        )  # (line_idx, canonical_label, heading_text)
        for i, line in enumerate(lines):
            canonical = rules.match_heading(line)
            if canonical:
                heading_positions.append((i, canonical, line.strip()))

        blocks: List[RawBlock] = []

        if not heading_positions:
            # No headings anywhere -> pure NLP fallback over the whole document.
            blocks.extend(_classify_unlabeled_span(lines, 0, is_leading=True))
            return self._attach_nlp_opinions(blocks)

        # 2. Leading span before the first heading -> header/contact/summary.
        first_heading_line = heading_positions[0][0]
        if first_heading_line > 0:
            leading_lines = lines[0:first_heading_line]
            blocks.extend(_classify_unlabeled_span(leading_lines, 0, is_leading=True))

        # 3. Slice content between each heading and the next.
        for idx, (line_idx, canonical, heading_text) in enumerate(heading_positions):
            content_start = line_idx + 1
            content_end = heading_positions[idx + 1][0] if idx + 1 < len(heading_positions) else n
            content_lines = lines[content_start:content_end]
            content_text = "\n".join(content_lines).strip("\n")

            if not content_text.strip():
                continue

            nlp_result = nlp_classifier.classify_block(content_text)
            block = RawBlock(
                label=canonical,
                text=content_text,
                method="rule_heading",
                confidence=1.0,
                start_line=content_start,
                end_line=content_end - 1,
                nlp_label=nlp_result.label,
                nlp_confidence=nlp_result.confidence,
                heading_text=heading_text,
            )
            if nlp_result.label != "Uncategorized" and nlp_result.label != canonical:
                block.flags.append(
                    f"heading says '{canonical}' but content reads as '{nlp_result.label}' "
                    f"(nlp confidence {nlp_result.confidence})"
                )
            blocks.append(block)

        return blocks

    @staticmethod
    def _attach_nlp_opinions(blocks: List[RawBlock]) -> List[RawBlock]:
        # In the no-heading path, nlp_label/nlp_confidence are already
        # set (the block *is* the NLP classification), so nothing to do.
        return blocks
