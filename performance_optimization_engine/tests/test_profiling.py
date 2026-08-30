import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from performance_optimization_engine.profiling import Timer, timed  # noqa: E402


def test_timer_measures_duration():
    with Timer("sleep_test") as t:
        time.sleep(0.01)
    assert t.metrics is not None
    assert t.metrics.duration_seconds >= 0.01
    assert t.metrics.operation == "sleep_test"


def test_timer_computes_throughput():
    with Timer("batch", records_processed=100) as t:
        time.sleep(0.01)
    assert t.metrics.throughput_per_sec > 0


def test_timer_zero_records_zero_throughput():
    with Timer("noop") as t:
        pass
    assert t.metrics.throughput_per_sec == 0.0


def test_metrics_to_dict_serializable():
    with Timer("op") as t:
        pass
    d = t.metrics.to_dict()
    assert "operation" in d
    assert "duration_seconds" in d


def test_timed_decorator_attaches_metrics():
    @timed("decorated_op")
    def work():
        time.sleep(0.005)
        return 42

    result = work()
    assert result == 42
    assert work.last_metrics is not None
    assert work.last_metrics.operation == "decorated_op"


def test_memory_source_reported():
    with Timer("mem_test") as t:
        _ = [x for x in range(1000)]
    assert t.metrics.memory_source in ("psutil", "tracemalloc", "unavailable")
