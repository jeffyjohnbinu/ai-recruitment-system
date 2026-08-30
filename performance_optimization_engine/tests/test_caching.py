import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from performance_optimization_engine.caching import (  # noqa: E402
    EmbeddingCache,
    ExtractionCache,
    LRUCache,
)


def test_lru_cache_basic_get_set():
    cache = LRUCache(max_size=2)
    cache.set("a", 1)
    assert cache.get("a") == 1


def test_lru_cache_miss_returns_none():
    cache = LRUCache()
    assert cache.get("missing") is None


def test_lru_cache_eviction():
    cache = LRUCache(max_size=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)  # evicts "a" (least recently used)
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_lru_cache_ttl_expiry():
    cache = LRUCache(max_size=10, ttl_seconds=0.01)
    cache.set("a", 1)
    time.sleep(0.02)
    assert cache.get("a") is None


def test_lru_cache_stats():
    cache = LRUCache(max_size=10)
    cache.set("a", 1)
    cache.get("a")
    cache.get("missing")
    stats = cache.stats
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_extraction_cache_hit_after_set(tmp_path):
    cache = ExtractionCache()
    f = tmp_path / "resume.txt"
    f.write_text("hello world")

    assert cache.get(f) is None
    cache.set(f, {"cleaned_text": "hello world"})
    assert cache.get(f) == {"cleaned_text": "hello world"}


def test_extraction_cache_invalidated_on_content_change(tmp_path):
    cache = ExtractionCache()
    f = tmp_path / "resume.txt"
    f.write_text("version one")
    cache.set(f, {"v": 1})

    time.sleep(0.01)
    f.write_text("version two, much longer content than before")
    assert cache.get(f) is None


def test_extraction_cache_missing_file_does_not_crash():
    cache = ExtractionCache()
    result = cache.get("/nonexistent/path/resume.pdf")
    assert result is None


def test_embedding_cache_accepts_string():
    cache = EmbeddingCache(model_version="v1")
    cache.set("some resume text", [0.1, 0.2])
    assert cache.get("some resume text") == [0.1, 0.2]


def test_embedding_cache_accepts_dict_alias_variants():
    cache = EmbeddingCache(model_version="v1")
    cache.set({"cleaned_text": "hello"}, [1, 2, 3])
    # Different field name but identical text content -> same cache key.
    assert cache.get({"text": "hello"}) == [1, 2, 3]
    assert cache.get({"cleaned_text": "hello"}) == [1, 2, 3]


def test_embedding_cache_normalizes_whitespace_and_case():
    cache = EmbeddingCache(model_version="v1")
    cache.set("Hello   World", [9])
    assert cache.get("hello world") == [9]


def test_embedding_cache_rejects_unsupported_type():
    cache = EmbeddingCache()
    with pytest.raises(TypeError):
        cache.get(12345)


def test_embedding_cache_different_model_versions_dont_collide():
    cache_a = EmbeddingCache(model_version="v1")
    cache_b = EmbeddingCache(model_version="v2")
    cache_a.set("text", [1])
    assert cache_b.get("text") is None
