"""
Parallel processing
--------------------
Speeds up batch operations (e.g. extracting a directory of hundreds of
resumes) by fanning work out across threads. Threads (not processes) are
used deliberately: the dominant cost in extraction is I/O (disk reads,
PDF/DOCX parsing libraries that release the GIL during C-extension calls),
so a ThreadPoolExecutor gives most of the available speedup without the
pickling overhead and memory duplication of a ProcessPoolExecutor.

`parallel_map` falls back to sequential execution automatically when
`max_workers=1` or the input is too small to benefit, so callers never
need to branch on batch size themselves.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, TypeVar

logger = logging.getLogger("performance_optimization_engine.parallel_processing")

T = TypeVar("T")
R = TypeVar("R")

_MIN_ITEMS_FOR_PARALLELISM = 2


@dataclass
class ParallelRunResult:
    """Aggregated outcome of a `parallel_map` call."""

    results: List[Any] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    worker_count_used: int = 1
    total_items: int = 0

    @property
    def success_count(self) -> int:
        return len(self.results)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_items": self.total_items,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "worker_count_used": self.worker_count_used,
            "errors": self.errors,
        }


def parallel_map(
    func: Callable[[T], R],
    items: Iterable[T],
    max_workers: int = 4,
    item_label: Callable[[T], str] = str,
) -> ParallelRunResult:
    """
    Apply `func` to every item, in parallel when it's worthwhile.

    Order of `results` is NOT guaranteed to match `items` order (results
    are collected as they complete) — callers that need positional
    correspondence should have `func` return an identifying key alongside
    its output.

    A per-item exception is caught and recorded in `errors` rather than
    aborting the whole batch, so one malformed resume doesn't take down
    processing of the other 499.
    """
    item_list = list(items)
    result = ParallelRunResult(total_items=len(item_list))

    if len(item_list) < _MIN_ITEMS_FOR_PARALLELISM or max_workers <= 1:
        result.worker_count_used = 1
        for item in item_list:
            try:
                result.results.append(func(item))
            except Exception as exc:  # noqa: BLE001
                logger.warning("parallel_map item failed: %s (%s)", item_label(item), exc)
                result.errors.append({"item": item_label(item), "error": str(exc)})
        return result

    workers = min(max_workers, len(item_list))
    result.worker_count_used = workers

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_item = {executor.submit(func, item): item for item in item_list}
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                result.results.append(future.result())
            except Exception as exc:  # noqa: BLE001
                logger.warning("parallel_map item failed: %s (%s)", item_label(item), exc)
                result.errors.append({"item": item_label(item), "error": str(exc)})

    return result


class BatchExtractor:
    """
    Thin parallel wrapper around any single-file processing callable
    (e.g. `ResumeExtractionEngine.process_file`). Kept decoupled from
    the Day 5 engine itself (duck-typed callable) so this module never
    imports — and therefore never needs to modify — upstream packages.
    """

    def __init__(self, process_file_fn: Callable[[Any], Any], max_workers: int = 4):
        self.process_file_fn = process_file_fn
        self.max_workers = max_workers

    def process_files(self, file_paths: Iterable[Any]) -> ParallelRunResult:
        return parallel_map(
            self.process_file_fn,
            file_paths,
            max_workers=self.max_workers,
            item_label=lambda p: str(p),
        )
