"""
Embedding Providers
--------------------
Turns free text (skills lists, experience summaries, project blurbs) into
fixed-length numeric vectors so semantic similarity can be measured with
cosine distance, instead of relying on exact keyword overlap.

Two engines, selected automatically at runtime:

  1. SentenceTransformerEmbedder (preferred) -- uses the `sentence-transformers`
     library (model: "all-MiniLM-L6-v2") for true semantic embeddings that
     understand paraphrase / synonym relationships ("led a team" ~ "managed
     engineers"). Requires the package to be installed AND the model weights
     to be reachable (first run downloads from Hugging Face).

  2. HashingBoWEmbedder (fallback) -- a dependency-light, fully offline
     embedder built on scikit-learn's HashingVectorizer + a light synonym
     expansion pass. Produces deterministic vectors with no network access
     and no corpus-fitting step required (unlike TF-IDF, which needs a
     background corpus to compute IDF weights meaningfully for a single
     resume/JD pair).

This mirrors the two-engine fallback pattern already used in
`resume_extraction_engine/readers/pdf_reader.py` (pdfplumber -> PyPDF2) and
keeps the matching engine usable even in network-restricted environments.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Protocol

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

logger = logging.getLogger("semantic_matching_engine.embeddings")

try:
    from sentence_transformers import SentenceTransformer

    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    _HAS_SENTENCE_TRANSFORMERS = False

_DEFAULT_ST_MODEL = "all-MiniLM-L6-v2"

# Small, hand-curated synonym map for common resume/JD terminology. This is
# NOT a replacement for real embeddings -- it just narrows the gap for the
# offline fallback engine so "lead" / "manage", "build" / "develop", etc.
# collide onto the same hashed feature bucket instead of being treated as
# unrelated tokens.
_SYNONYM_GROUPS: List[List[str]] = [
    ["lead", "led", "manage", "managed", "oversee", "oversaw", "supervise", "supervised"],
    ["build", "built", "develop", "developed", "create", "created", "implement", "implemented"],
    ["design", "designed", "architect", "architected"],
    ["improve", "improved", "optimize", "optimized", "enhance", "enhanced"],
    ["reduce", "reduced", "cut", "decrease", "decreased", "lower", "lowered"],
    ["increase", "increased", "boost", "boosted", "grow", "grew"],
    ["team", "teams", "engineers", "developers", "staff"],
    ["backend", "back-end", "server-side"],
    ["frontend", "front-end", "client-side"],
    ["ml", "machine learning", "ai", "artificial intelligence"],
    ["db", "database", "databases"],
]

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+#.]{1,}")


def _build_synonym_lookup() -> dict:
    lookup = {}
    for group in _SYNONYM_GROUPS:
        canonical = group[0]
        for term in group:
            lookup[term] = canonical
    return lookup


_SYNONYM_LOOKUP = _build_synonym_lookup()


def _canonicalize_tokens(text: str) -> str:
    """Lowercase, tokenize, and collapse known synonyms onto one canonical
    token so the hashing vectorizer treats them as (partially) the same
    signal. Multi-word synonyms (e.g. "machine learning") are handled via
    a direct substring pass before tokenization."""
    lowered = text.lower()
    for group in _SYNONYM_GROUPS:
        canonical = group[0]
        for term in sorted(group, key=len, reverse=True):
            if " " in term:
                lowered = lowered.replace(term, canonical)
    tokens = _TOKEN_RE.findall(lowered)
    canon_tokens = [_SYNONYM_LOOKUP.get(t, t) for t in tokens]
    return " ".join(canon_tokens)


class Embedder(Protocol):
    """Common interface both embedding engines implement."""

    engine_name: str

    def embed(self, texts: List[str]) -> np.ndarray: ...


@dataclass
class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model. Loaded lazily so importing this
    module never triggers a network call or heavy model load by itself."""

    model_name: str = _DEFAULT_ST_MODEL
    engine_name: str = "sentence-transformers"
    _model: Optional["SentenceTransformer"] = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            logger.info("Loading sentence-transformers model '%s'", self.model_name)
            self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        self._ensure_loaded()
        vectors = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float64)


@dataclass
class HashingBoWEmbedder:
    """Offline, dependency-light fallback embedder. Not a substitute for
    real semantic embeddings, but degrades gracefully instead of crashing
    when sentence-transformers / network access is unavailable."""

    n_features: int = 2**14
    engine_name: str = "hashing-bow-fallback"

    def __post_init__(self) -> None:
        self._vectorizer = HashingVectorizer(
            n_features=self.n_features,
            alternate_sign=False,
            norm="l2",
            preprocessor=_canonicalize_tokens,
        )

    def embed(self, texts: List[str]) -> np.ndarray:
        matrix = self._vectorizer.transform(texts)
        return matrix.toarray().astype(np.float64)


def get_embedder(prefer: str = "auto") -> Embedder:
    """
    Select the best available embedding engine.

    prefer:
      "auto"                 -> try sentence-transformers, fall back silently
      "sentence-transformers" -> force it (raises if unavailable)
      "hashing"               -> force the offline fallback
    """
    if prefer == "hashing":
        return HashingBoWEmbedder()

    if prefer == "sentence-transformers":
        if not _HAS_SENTENCE_TRANSFORMERS:
            raise RuntimeError("sentence-transformers is not installed.")
        return SentenceTransformerEmbedder()

    # auto
    if _HAS_SENTENCE_TRANSFORMERS:
        try:
            embedder = SentenceTransformerEmbedder()
            embedder._ensure_loaded()  # eagerly probe; raises if model can't be fetched
            logger.info("Using sentence-transformers embedder.")
            return embedder
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "sentence-transformers unavailable at runtime (%s). "
                "Falling back to offline hashing embedder.",
                exc,
            )

    logger.info("Using offline hashing-bow fallback embedder.")
    return HashingBoWEmbedder()
