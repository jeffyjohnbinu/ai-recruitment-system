"""
NLP-based section classifier
------------------------------
For text blocks that have no explicit heading (a resume with no
section labels at all, a stray paragraph, or a table row dropped
into the wrong place), fall back to a lightweight, dependency-free
keyword-weighted scorer instead of requiring an explicit heading.

This is intentionally *not* a deep-learning classifier — the project
has no network access to download embedding/spaCy models inside the
build sandbox, and a transparent, inspectable scorer is easier to
validate and extend than a black-box one for a first cut. Each
canonical label has a lexicon of weighted keyword/phrase signals;
a block's score for a label is the weighted keyword-hit density
(weighted hits / word count). A couple of structural signals (date
ranges, email/phone patterns, bullet density) supplement the raw
keyword hits, since those are strong, cheap signals for Experience
vs. Education vs. Contact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

_DATE_RANGE_RE = re.compile(
    r"\b(19|20)\d{2}\s*[-–—to]{1,3}\s*((19|20)\d{2}|present|current)\b", re.IGNORECASE
)
_DEGREE_RE = re.compile(
    r"\b(b\.?s\.?|b\.?a\.?|m\.?s\.?|m\.?a\.?|m\.?b\.?a\.?|ph\.?d\.?|bachelor|master|"
    r"associate degree|doctorate)\b",
    re.IGNORECASE,
)
_BULLET_RE = re.compile(r"^\s*-\s+")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"(\+?\d[\d\-\s()]{6,}\d)")

# label -> list of (keyword/phrase, weight). Multi-word phrases are
# matched as substrings (case-insensitive); single words are matched
# on word boundaries so "AI" doesn't match inside "email".
_LEXICON: Dict[str, List[Tuple[str, float]]] = {
    "Skills": [
        ("python", 2),
        ("java", 1.5),
        ("sql", 2),
        ("aws", 2),
        ("azure", 2),
        ("gcp", 1.5),
        ("docker", 2),
        ("kubernetes", 2),
        ("javascript", 1.5),
        ("react", 1.5),
        ("node", 1.2),
        ("proficient", 2),
        ("proficiency", 1.5),
        ("tools", 1.2),
        ("technologies", 1.5),
        ("frameworks", 1.5),
        ("programming languages", 2.5),
        ("technical skills", 3),
        ("software", 1),
        ("cloud", 1.2),
        ("api", 1),
        ("rest apis", 1.5),
        ("ci/cd", 1.5),
        ("excel", 1.2),
        ("tableau", 1.5),
        ("git", 1),
        ("linux", 1.2),
        ("machine learning", 1.5),
        ("competencies", 1.5),
        ("expertise", 1.2),
        ("figma", 1.5),
        ("photoshop", 1.5),
    ],
    "Experience": [
        ("managed", 1.5),
        ("led", 1.2),
        ("developed", 1.2),
        ("built", 1),
        ("designed", 1),
        ("responsible for", 2),
        ("achieved", 1.5),
        ("increased", 1.5),
        ("reduced", 1.5),
        ("engineer", 1.2),
        ("manager", 1.2),
        ("company", 1),
        ("present", 1),
        ("mentored", 1.5),
        ("shipped", 1.5),
        ("owned", 1.2),
        ("partnered", 1.2),
        ("collaborated", 1.2),
        ("deployed", 1.2),
        ("implemented", 1.2),
        ("spearheaded", 1.8),
        ("years of experience", 2.5),
        ("role", 1),
    ],
    "Education": [
        ("university", 2.5),
        ("college", 2),
        ("gpa", 2),
        ("graduated", 2),
        ("degree", 2),
        ("major", 1.2),
        ("coursework", 1.5),
        ("thesis", 1.5),
        ("academic", 1.2),
        ("school of", 1.5),
        ("institute of technology", 2),
    ],
    "Certifications": [
        ("certified", 2.5),
        ("certification", 2.5),
        ("license", 1.5),
        ("credential", 1.5),
        ("pmp", 2),
        ("cfa", 2),
        ("cpa", 2),
        ("scrum master", 2),
        ("issued", 1.2),
    ],
    "Projects": [
        ("project", 2),
        ("github", 2),
        ("personal project", 2.5),
        ("capstone", 2),
        ("built a", 1.5),
        ("developed a", 1.5),
        ("side project", 2),
        ("open source", 1.5),
        ("repository", 1.2),
        ("demo", 1),
    ],
    "Summary": [
        ("results-driven", 2),
        ("passionate", 1.5),
        ("seeking", 1.5),
        ("summary", 2),
        ("objective", 2),
        ("motivated", 1.2),
        ("dedicated", 1.2),
        ("years of experience", 1.2),
        ("proven track record", 2),
    ],
    "Achievements": [
        ("award", 2.5),
        ("achievement", 2),
        ("honor", 2),
        ("recognized", 1.8),
        ("winner", 2),
        ("scholarship", 2),
        ("top performer", 2),
    ],
    "Languages": [
        ("fluent", 2.5),
        ("native speaker", 2.5),
        ("language proficiency", 2.5),
        ("spanish", 1.2),
        ("french", 1.2),
        ("german", 1.2),
        ("mandarin", 1.2),
        ("bilingual", 2),
    ],
    "Contact": [
        ("linkedin", 2),
        ("github.com", 1.5),
        ("portfolio", 1.2),
        ("address", 1.2),
    ],
}

_MIN_CONFIDENCE = 0.06  # below this, label as "Uncategorized" rather than guess


@dataclass
class NLPClassification:
    label: str
    confidence: float
    scores: Dict[str, float]


def _word_count(text: str) -> int:
    return max(1, len(re.findall(r"\w+", text)))


def _keyword_score(text_lower: str, word_count: int) -> Dict[str, float]:
    scores: Dict[str, float] = {label: 0.0 for label in _LEXICON}
    for label, lexicon in _LEXICON.items():
        total = 0.0
        for phrase, weight in lexicon:
            if " " in phrase or "/" in phrase:
                hits = text_lower.count(phrase)
            else:
                hits = len(re.findall(rf"\b{re.escape(phrase)}\b", text_lower))
            total += hits * weight
        scores[label] = total / word_count
    return scores


def classify_block(text: str) -> NLPClassification:
    """
    Score a block of text against every canonical label's keyword
    lexicon plus a few structural heuristics, and return the
    best-scoring label with a normalized confidence in [0, 1].
    """
    text_lower = text.lower()
    word_count = _word_count(text)
    scores = _keyword_score(text_lower, word_count)

    # Structural boosts -------------------------------------------------
    has_date_range = bool(_DATE_RANGE_RE.search(text_lower))
    has_degree = bool(_DEGREE_RE.search(text_lower))
    bullet_ratio = sum(1 for ln in text.splitlines() if _BULLET_RE.match(ln)) / max(
        1, len(text.splitlines())
    )
    has_contact = bool(_EMAIL_RE.search(text)) or bool(_PHONE_RE.search(text))

    if has_degree:
        scores["Education"] += 1.5
    if has_date_range and not has_degree:
        scores["Experience"] += 0.8
    if bullet_ratio > 0.4:
        scores["Experience"] += 0.4
        scores["Skills"] += 0.1
    if has_contact and word_count < 25:
        scores["Contact"] += 1.5

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    # Convert to a bounded, human-readable confidence value. Divisor
    # tuned against the fixture corpus (tests/fixtures/) so that a
    # single strong keyword hit in a short block is enough to clear
    # the classification threshold, without letting one-off word
    # matches in an otherwise unrelated block win.
    confidence = min(1.0, best_score / 1.5)

    if confidence < _MIN_CONFIDENCE:
        return NLPClassification(
            label="Uncategorized", confidence=round(confidence, 3), scores=scores
        )

    return NLPClassification(label=best_label, confidence=round(confidence, 3), scores=scores)
