import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from performance_optimization_engine.memory_optimizer import (  # noqa: E402
    MemoryBudgetExceededError,
    MemoryBudgetGuard,
    chunked,
    iter_text_pages,
)


def test_chunked_groups_correctly():
    chunks = list(chunked(range(10), 3))
    assert chunks == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


def test_chunked_exact_multiple():
    chunks = list(chunked(range(6), 3))
    assert chunks == [[0, 1, 2], [3, 4, 5]]


def test_chunked_rejects_nonpositive_size():
    with pytest.raises(ValueError):
        list(chunked(range(5), 0))


def test_chunked_empty_input():
    assert list(chunked([], 5)) == []


def test_iter_text_pages_short_text_single_page():
    pages = list(iter_text_pages("short text", page_char_size=1000))
    assert pages == ["short text"]


def test_iter_text_pages_empty_text_no_pages():
    assert list(iter_text_pages("", page_char_size=100)) == []


def test_iter_text_pages_splits_long_text():
    long_text = "\n\n".join([f"Paragraph {i} " + "x" * 50 for i in range(20)])
    pages = list(iter_text_pages(long_text, page_char_size=200))
    assert len(pages) > 1
    assert all(len(p) <= 250 for p in pages)  # some slack for paragraph joins


def test_iter_text_pages_rejects_nonpositive_size():
    with pytest.raises(ValueError):
        list(iter_text_pages("text", page_char_size=0))


def test_iter_text_pages_hard_splits_oversized_paragraph():
    huge_paragraph = "y" * 5000
    pages = list(iter_text_pages(huge_paragraph, page_char_size=1000))
    assert len(pages) == 5
    assert all(len(p) <= 1000 for p in pages)


def test_memory_budget_guard_checkpoint_runs_without_error():
    with MemoryBudgetGuard(budget_mb=100000, raise_on_exceed=False) as guard:
        guard.checkpoint()
    assert guard is not None


def test_memory_budget_guard_no_psutil_is_noop(monkeypatch):
    import performance_optimization_engine.memory_optimizer as mod

    monkeypatch.setattr(mod, "_HAS_PSUTIL", False)
    guard = mod.MemoryBudgetGuard(budget_mb=1)
    result = guard.checkpoint()
    assert result is None


def test_memory_budget_guard_raises_when_exceeded(monkeypatch):
    import performance_optimization_engine.memory_optimizer as mod

    class FakeProcess:
        def memory_info(self):
            class Info:
                rss = 500 * 1024 * 1024  # 500MB

            return Info()

    guard = mod.MemoryBudgetGuard(budget_mb=100, raise_on_exceed=True)
    guard._process = FakeProcess()
    with pytest.raises(MemoryBudgetExceededError):
        guard.checkpoint()
