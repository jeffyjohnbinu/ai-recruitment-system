"""
Entity detection refinement
-----------------------------
Post-processes entity/skill extraction output (Day 9 skill_extraction_engine,
Day 10 experience_parsing_engine, Day 11 education_certification_extractor)
to improve precision without touching the upstream extractors:

  - deduplicates near-duplicate entities ("Node.js" / "NodeJS" / "node js")
  - merges confidence scores for the same canonical entity found via
    multiple extraction passes (rule-based + fuzzy)
  - drops low-confidence noise below a configurable threshold
  - canonicalizes casing/punctuation for consistent downstream matching

Two-pass pattern: uses `rapidfuzz` for fuzzy dedup when available, and
falls back to Python's stdlib `difflib` (slower, slightly less accurate)
when it isn't — with a logged warning, consistent with every other
fallback in this repository.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List

logger = logging.getLogger("performance_optimization_engine.entity_refinement")

try:
    from rapidfuzz import fuzz as _rapidfuzz_fuzz

    _HAS_RAPIDFUZZ = True
except ImportError:  # pragma: no cover
    _HAS_RAPIDFUZZ = False
    logger.warning(
        "rapidfuzz not installed — entity_refinement falling back to stdlib "
        "difflib for fuzzy matching (slower, slightly less accurate)."
    )

_DEFAULT_SIMILARITY_THRESHOLD = 88.0  # 0-100 scale, matches rapidfuzz convention
_DEFAULT_MIN_CONFIDENCE = 0.35
_NOISE_PATTERN = re.compile(r"^[\W_]+$")


@dataclass
class RefinedEntity:
    canonical_name: str
    confidence: float
    source_variants: List[str] = field(default_factory=list)
    merge_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical_name": self.canonical_name,
            "confidence": round(self.confidence, 4),
            "source_variants": self.source_variants,
            "merge_count": self.merge_count,
        }


@dataclass
class RefinementResult:
    entities: List[RefinedEntity]
    input_count: int
    output_count: int
    dropped_low_confidence: int
    merged_duplicates: int
    similarity_backend: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "input_count": self.input_count,
            "output_count": self.output_count,
            "dropped_low_confidence": self.dropped_low_confidence,
            "merged_duplicates": self.merged_duplicates,
            "similarity_backend": self.similarity_backend,
        }


def _similarity(a: str, b: str) -> float:
    """Return a 0-100 similarity score, via rapidfuzz if available."""
    if _HAS_RAPIDFUZZ:
        return float(_rapidfuzz_fuzz.token_sort_ratio(a, b))
    return SequenceMatcher(None, a, b).ratio() * 100.0


def _normalize_name(name: str) -> str:
    cleaned = re.sub(r"[._/]+", " ", name.strip().lower())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _tight_normalize_name(name: str) -> str:
    """Whitespace-insensitive normalization used for similarity comparison,
    so "Node.js" / "NodeJS" / "node js" all collapse to the same key even
    though their separator characters differ."""
    return re.sub(r"[\s._/]+", "", name.strip().lower())


def _canonical_casing(name: str) -> str:
    """
    Title-case each token but preserve short all-caps acronyms (AWS, SQL,
    CI/CD-style tokens) rather than lower-casing them, mirroring the
    Day 5 TextNormalizer's acronym-preservation rule for consistency
    across the codebase.
    """
    tokens = name.split()
    out = []
    for tok in tokens:
        core = re.sub(r"[^A-Za-z0-9]", "", tok)
        if core.isupper() and 2 <= len(core) <= 5:
            out.append(tok)
        elif any(ch.isdigit() for ch in tok):
            out.append(tok)  # e.g. "Python3", "C++11" — leave alphanumerics alone
        else:
            out.append(tok.capitalize())
    return " ".join(out)


def _extract_entity_list(source: Any) -> List[Dict[str, Any]]:
    """
    Alias-tolerant ingestion: accepts a plain list of strings, a list of
    dicts with any of "name"/"skill"/"entity"/"text" and optional
    "confidence"/"score", or a dict with an "entities"/"skills" key
    wrapping such a list — matching the shape variance seen across
    Day 9-11 outputs without requiring their schemas to change.
    """
    if isinstance(source, dict):
        for key in ("entities", "skills", "items", "results"):
            if key in source and isinstance(source[key], list):
                source = source[key]
                break
        else:
            raise ValueError("Could not find an entity list in the provided dict.")

    if not isinstance(source, list):
        raise TypeError("Entity source must be a list or a dict wrapping a list.")

    normalized: List[Dict[str, Any]] = []
    for item in source:
        if isinstance(item, str):
            normalized.append({"name": item, "confidence": 1.0})
        elif isinstance(item, dict):
            name = None
            for key in ("name", "skill", "entity", "text", "value"):
                if key in item and isinstance(item[key], str):
                    name = item[key]
                    break
            if name is None:
                continue
            confidence = 1.0
            for key in ("confidence", "score", "match_confidence"):
                if key in item and isinstance(item[key], (int, float)):
                    confidence = float(item[key])
                    break
            normalized.append({"name": name, "confidence": confidence})
    return normalized


class EntityRefiner:
    """Refines a raw entity/skill list into a deduplicated, confidence-scored set."""

    def __init__(
        self,
        similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
    ):
        self.similarity_threshold = similarity_threshold
        self.min_confidence = min_confidence

    def refine(self, source: Any) -> RefinementResult:
        raw_entities = _extract_entity_list(source)
        input_count = len(raw_entities)

        # Drop pure-noise / empty tokens up front.
        candidates = [
            e
            for e in raw_entities
            if e["name"].strip() and not _NOISE_PATTERN.match(e["name"].strip())
        ]

        clusters: List[List[Dict[str, Any]]] = []
        for entity in candidates:
            tight = _tight_normalize_name(entity["name"])
            placed = False
            for cluster in clusters:
                cluster_tight = _tight_normalize_name(cluster[0]["name"])
                if (
                    tight == cluster_tight
                    or _similarity(tight, cluster_tight) >= self.similarity_threshold
                ):
                    cluster.append(entity)
                    placed = True
                    break
            if not placed:
                clusters.append([entity])

        merged_duplicates = sum(len(c) - 1 for c in clusters if len(c) > 1)

        refined: List[RefinedEntity] = []
        dropped = 0
        for cluster in clusters:
            best = max(cluster, key=lambda e: e["confidence"])
            avg_confidence = sum(e["confidence"] for e in cluster) / len(cluster)
            # Reward entities confirmed by multiple extraction passes.
            boosted_confidence = min(1.0, avg_confidence + 0.05 * (len(cluster) - 1))

            if boosted_confidence < self.min_confidence:
                dropped += 1
                continue

            refined.append(
                RefinedEntity(
                    canonical_name=_canonical_casing(best["name"].strip()),
                    confidence=boosted_confidence,
                    source_variants=sorted({e["name"] for e in cluster}),
                    merge_count=len(cluster),
                )
            )

        refined.sort(key=lambda e: e.confidence, reverse=True)

        return RefinementResult(
            entities=refined,
            input_count=input_count,
            output_count=len(refined),
            dropped_low_confidence=dropped,
            merged_duplicates=merged_duplicates,
            similarity_backend="rapidfuzz" if _HAS_RAPIDFUZZ else "difflib",
        )
