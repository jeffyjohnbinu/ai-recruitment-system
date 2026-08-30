"""
Memory handling
----------------
Utilities that keep peak memory bounded when processing large batches
of resumes or long documents:

  - `chunked`            : generic generator chunking for streaming
                            batch processing instead of loading
                            everything into one list.
  - `iter_text_pages`     : splits a large cleaned-text blob into
                            page-sized windows for streaming downstream
                            NLP instead of holding the whole document
                            (and every intermediate copy) in memory.
  - `MemoryBudgetGuard`   : context manager that warns (and can raise)
                            when resident memory crosses a configured
                            budget, so a runaway batch job fails fast
                            in CI instead of OOM-killing the box.
"""

from __future__ import annotations

import logging
from typing import Iterable, Iterator, List, Optional, TypeVar

logger = logging.getLogger("performance_optimization_engine.memory_optimizer")

try:
    import psutil  # type: ignore

    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover
    _HAS_PSUTIL = False

T = TypeVar("T")

_DEFAULT_PAGE_CHAR_SIZE = 4000


def chunked(items: Iterable[T], size: int) -> Iterator[List[T]]:
    """
    Yield successive chunks of `size` from `items` without materializing
    the whole input as one list. Used to bound the number of resumes
    held in memory simultaneously during batch extraction/scoring.
    """
    if size <= 0:
        raise ValueError("chunk size must be positive")

    chunk: List[T] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def iter_text_pages(text: str, page_char_size: int = _DEFAULT_PAGE_CHAR_SIZE) -> Iterator[str]:
    """
    Split large cleaned resume/JD text into fixed-size windows on
    paragraph boundaries where possible, so a downstream NLP call
    (skill extraction, embedding) can stream a long document instead of
    tokenizing the whole thing — and its intermediate copies — at once.
    """
    if page_char_size <= 0:
        raise ValueError("page_char_size must be positive")

    if len(text) <= page_char_size:
        if text:
            yield text
        return

    paragraphs = text.split("\n\n")
    buffer = ""
    for para in paragraphs:
        candidate = f"{buffer}\n\n{para}" if buffer else para
        if len(candidate) > page_char_size and buffer:
            yield buffer
            buffer = para
        else:
            buffer = candidate

        # A single paragraph longer than the page size on its own still
        # needs hard-splitting so we never yield an unbounded page.
        while len(buffer) > page_char_size:
            yield buffer[:page_char_size]
            buffer = buffer[page_char_size:]

    if buffer:
        yield buffer


class MemoryBudgetExceededError(RuntimeError):
    pass


class MemoryBudgetGuard:
    """
    Context manager that checks resident memory against `budget_mb` on
    exit (and optionally at explicit `.checkpoint()` calls during a long
    batch loop). Degrades to a no-op with a logged warning when psutil
    isn't installed, since memory introspection without it is unreliable
    cross-platform.

    Usage:
        with MemoryBudgetGuard(budget_mb=1500, raise_on_exceed=False) as guard:
            for batch in chunked(files, 50):
                process(batch)
                guard.checkpoint()
    """

    def __init__(self, budget_mb: Optional[float] = None, raise_on_exceed: bool = False):
        self.budget_mb = budget_mb
        self.raise_on_exceed = raise_on_exceed
        self.peak_mb: Optional[float] = None
        self._process = None

        if _HAS_PSUTIL:
            self._process = psutil.Process()
        else:
            logger.warning(
                "psutil not installed — MemoryBudgetGuard cannot measure RSS; "
                "budget checks are no-ops."
            )

    def _current_rss_mb(self) -> Optional[float]:
        if self._process is None:
            return None
        try:
            return self._process.memory_info().rss / (1024 * 1024)
        except Exception:  # noqa: BLE001
            return None

    def checkpoint(self) -> Optional[float]:
        current = self._current_rss_mb()
        if current is None:
            return None
        if self.peak_mb is None or current > self.peak_mb:
            self.peak_mb = current
        if self.budget_mb is not None and current > self.budget_mb:
            message = f"Memory budget exceeded: {current:.1f}MB > {self.budget_mb:.1f}MB"
            if self.raise_on_exceed:
                raise MemoryBudgetExceededError(message)
            logger.warning(message)
        return current

    def __enter__(self) -> "MemoryBudgetGuard":
        self.checkpoint()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.checkpoint()
