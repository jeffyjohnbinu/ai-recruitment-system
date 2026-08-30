"""
Caching layer
-------------
Reduces redundant work (and therefore model response time / extraction
time) with two complementary caches:

  - `LRUCache`   : generic in-memory least-recently-used cache with
                   optional TTL, used as the base for both caches below.
  - `ExtractionCache` : keyed on (file content hash, mtime) so re-running
                   extraction on an unchanged resume is a cache hit.
  - `EmbeddingCache`  : keyed on (model_version, normalized text hash) so
                   repeated semantic-matching calls against the same
                   resume/JD text skip the embedding model entirely.

Both caches are alias-tolerant: they accept whatever key-shaped input
upstream modules already produce (dicts with "text"/"content", or raw
strings) rather than requiring a rigid schema.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("performance_optimization_engine.caching")


class LRUCache:
    """Simple thread-unsafe LRU cache with optional per-entry TTL (seconds)."""

    def __init__(self, max_size: int = 256, ttl_seconds: Optional[float] = None):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._store: "OrderedDict[str, tuple[Any, float]]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None

        value, stored_at = entry
        if self.ttl_seconds is not None and (time.time() - stored_at) > self.ttl_seconds:
            del self._store[key]
            self.misses += 1
            return None

        self._store.move_to_end(key)
        self.hits += 1
        return value

    def set(self, key: str, value: Any) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (value, time.time())
        if len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0

    @property
    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "size": len(self._store),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


class ExtractionCache:
    """
    Caches resume extraction output keyed on file content hash + mtime,
    so `ResumeExtractionEngine.process_file` results can be reused across
    repeated runs (e.g. CI re-runs, batch re-processing) without re-reading
    and re-cleaning an unchanged file.
    """

    def __init__(self, max_size: int = 512):
        self._cache = LRUCache(max_size=max_size)

    @staticmethod
    def _file_key(path: str | Path) -> str:
        p = Path(path)
        try:
            stat = p.stat()
            fingerprint = f"{p.resolve()}::{stat.st_size}::{stat.st_mtime_ns}"
        except FileNotFoundError:
            fingerprint = f"{p}::missing"
        return _hash_text(fingerprint)

    def get(self, path: str | Path) -> Optional[Any]:
        key = self._file_key(path)
        hit = self._cache.get(key)
        if hit is not None:
            logger.debug("Extraction cache hit for %s", path)
        return hit

    def set(self, path: str | Path, record: Any) -> None:
        key = self._file_key(path)
        self._cache.set(key, record)

    @property
    def stats(self) -> Dict[str, Any]:
        return self._cache.stats


class EmbeddingCache:
    """
    Caches embedding vectors (or any semantic-matching intermediate
    result) keyed on model version + normalized text, so repeated
    matching calls for the same candidate/job text pair skip the
    (comparatively expensive) sentence-transformers forward pass.

    Accepts alias-tolerant input: a raw string, or a dict containing
    any of "text", "content", "cleaned_text".
    """

    def __init__(self, max_size: int = 2048, model_version: str = "unknown"):
        self._cache = LRUCache(max_size=max_size)
        self.model_version = model_version

    @staticmethod
    def _extract_text(item: Any) -> str:
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            for key in ("text", "content", "cleaned_text", "raw_text"):
                if key in item and isinstance(item[key], str):
                    return item[key]
        raise TypeError(
            "EmbeddingCache expects a string or a dict with a text/content/"
            "cleaned_text/raw_text field."
        )

    def _key(self, item: Any) -> str:
        text = self._extract_text(item)
        normalized = " ".join(text.split()).lower()
        return _hash_text(f"{self.model_version}::{normalized}")

    def get(self, item: Any) -> Optional[Any]:
        return self._cache.get(self._key(item))

    def set(self, item: Any, embedding: Any) -> None:
        self._cache.set(self._key(item), embedding)

    @property
    def stats(self) -> Dict[str, Any]:
        return self._cache.stats
