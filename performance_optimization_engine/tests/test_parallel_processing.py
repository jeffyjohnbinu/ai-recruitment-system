import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from performance_optimization_engine.parallel_processing import (  # noqa: E402
    BatchExtractor,
    parallel_map,
)


def test_parallel_map_returns_all_results():
    result = parallel_map(lambda x: x * 2, [1, 2, 3, 4, 5], max_workers=4)
    assert sorted(result.results) == [2, 4, 6, 8, 10]
    assert result.total_items == 5
    assert result.error_count == 0


def test_parallel_map_falls_back_to_sequential_for_single_item():
    result = parallel_map(lambda x: x + 1, [5], max_workers=4)
    assert result.results == [6]
    assert result.worker_count_used == 1


def test_parallel_map_falls_back_when_max_workers_is_one():
    result = parallel_map(lambda x: x, [1, 2, 3], max_workers=1)
    assert result.worker_count_used == 1
    assert sorted(result.results) == [1, 2, 3]


def test_parallel_map_captures_per_item_errors():
    def maybe_fail(x):
        if x == 3:
            raise ValueError("boom")
        return x

    result = parallel_map(maybe_fail, [1, 2, 3, 4], max_workers=4)
    assert result.error_count == 1
    assert result.success_count == 3
    assert result.errors[0]["item"] == "3"


def test_parallel_map_empty_input():
    result = parallel_map(lambda x: x, [], max_workers=4)
    assert result.results == []
    assert result.total_items == 0


def test_batch_extractor_wraps_callable():
    calls = []

    def process_file(path):
        calls.append(path)
        return {"path": path, "status": "success"}

    extractor = BatchExtractor(process_file, max_workers=3)
    result = extractor.process_files(["a.pdf", "b.pdf", "c.docx"])
    assert result.success_count == 3
    assert len(calls) == 3


def test_result_to_dict_shape():
    result = parallel_map(lambda x: x, [1, 2], max_workers=2)
    d = result.to_dict()
    assert "total_items" in d
    assert "success_count" in d
    assert "worker_count_used" in d
