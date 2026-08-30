"""
Profiling utilities
--------------------
Lightweight timing and memory instrumentation used across every
optimization module. No hard dependency on `psutil` — degrades
gracefully to `tracemalloc` / `resource` when it isn't installed, and
to wall-clock-only measurement if neither is available.
"""

from __future__ import annotations

import logging
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("performance_optimization_engine.profiling")

try:
    import psutil  # type: ignore

    _PROCESS = psutil.Process()
    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover
    _HAS_PSUTIL = False
    _PROCESS = None


@dataclass
class PerformanceMetrics:
    """Result of profiling a single operation."""

    operation: str
    duration_seconds: float
    peak_memory_mb: Optional[float]
    memory_source: str  # "psutil" | "tracemalloc" | "unavailable"
    records_processed: int = 0
    throughput_per_sec: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "duration_seconds": round(self.duration_seconds, 6),
            "peak_memory_mb": (
                round(self.peak_memory_mb, 3) if self.peak_memory_mb is not None else None
            ),
            "memory_source": self.memory_source,
            "records_processed": self.records_processed,
            "throughput_per_sec": round(self.throughput_per_sec, 3),
            "extra": self.extra,
        }


class Timer:
    """
    Context manager measuring wall-clock duration and peak memory delta
    for a block of code.

    Usage:
        with Timer("extract_batch", records_processed=len(files)) as t:
            do_work()
        metrics = t.metrics
    """

    def __init__(self, operation: str, records_processed: int = 0, use_tracemalloc: bool = True):
        self.operation = operation
        self.records_processed = records_processed
        self.use_tracemalloc = use_tracemalloc
        self.metrics: Optional[PerformanceMetrics] = None
        self._start_time = 0.0
        self._start_rss_mb = None
        self._tracemalloc_started_here = False

    def __enter__(self) -> "Timer":
        self._start_time = time.perf_counter()
        if _HAS_PSUTIL:
            try:
                self._start_rss_mb = _PROCESS.memory_info().rss / (1024 * 1024)
            except Exception:  # noqa: BLE001
                self._start_rss_mb = None
        elif self.use_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start()
            self._tracemalloc_started_here = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        duration = time.perf_counter() - self._start_time

        peak_memory_mb: Optional[float] = None
        memory_source = "unavailable"

        if _HAS_PSUTIL and self._start_rss_mb is not None:
            try:
                end_rss_mb = _PROCESS.memory_info().rss / (1024 * 1024)
                peak_memory_mb = max(end_rss_mb, self._start_rss_mb)
                memory_source = "psutil"
            except Exception:  # noqa: BLE001
                pass
        elif tracemalloc.is_tracing():
            _, peak = tracemalloc.get_traced_memory()
            peak_memory_mb = peak / (1024 * 1024)
            memory_source = "tracemalloc"
            if self._tracemalloc_started_here:
                tracemalloc.stop()

        throughput = (
            self.records_processed / duration if duration > 0 and self.records_processed else 0.0
        )

        self.metrics = PerformanceMetrics(
            operation=self.operation,
            duration_seconds=duration,
            peak_memory_mb=peak_memory_mb,
            memory_source=memory_source,
            records_processed=self.records_processed,
            throughput_per_sec=throughput,
        )
        logger.info(
            "profiled operation=%s duration=%.4fs memory=%s(%s) throughput=%.2f/s",
            self.operation,
            duration,
            f"{peak_memory_mb:.2f}MB" if peak_memory_mb is not None else "n/a",
            memory_source,
            throughput,
        )


def timed(operation: Optional[str] = None) -> Callable:
    """Decorator form of `Timer` for functions that return a value plus a
    count of records processed isn't known ahead of time — wraps the call
    and attaches `.last_metrics` to the function object."""

    def decorator(func: Callable) -> Callable:
        op_name = operation or func.__name__

        def wrapper(*args, **kwargs):
            with Timer(op_name) as t:
                result = func(*args, **kwargs)
            wrapper.last_metrics = t.metrics
            return result

        wrapper.last_metrics = None
        return wrapper

    return decorator
