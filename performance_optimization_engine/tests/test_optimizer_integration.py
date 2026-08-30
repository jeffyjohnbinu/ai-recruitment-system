import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from performance_optimization_engine.optimizer import PerformanceOptimizationEngine  # noqa: E402


def make_engine(tmp_path):
    return PerformanceOptimizationEngine(output_dir=tmp_path, max_workers=3)


def test_optimize_extraction_processes_all_files_first_run(tmp_path):
    engine = make_engine(tmp_path)
    files = []
    for i in range(4):
        f = tmp_path / f"resume_{i}.txt"
        f.write_text(f"resume content {i}")
        files.append(f)

    calls = []

    def process(path):
        calls.append(path)
        return {"path": str(path), "status": "success"}

    result = engine.optimize_extraction(process, files)
    assert result.success_count == 4
    assert len(calls) == 4


def test_optimize_extraction_uses_cache_on_second_run(tmp_path):
    engine = make_engine(tmp_path)
    f = tmp_path / "resume.txt"
    f.write_text("resume content")

    calls = []

    def process(path):
        calls.append(path)
        return {"path": str(path), "status": "success"}

    engine.optimize_extraction(process, [f])
    engine.optimize_extraction(process, [f])  # second call should be a cache hit

    assert len(calls) == 1  # process() only invoked once across both runs


def test_optimize_extraction_reports_errors_without_aborting(tmp_path):
    engine = make_engine(tmp_path)
    f1 = tmp_path / "good.txt"
    f1.write_text("ok")
    f2 = tmp_path / "bad.txt"
    f2.write_text("bad")

    def process(path):
        if "bad" in str(path):
            raise ValueError("simulated failure")
        return {"status": "success"}

    result = engine.optimize_extraction(process, [f1, f2])
    assert result.success_count == 1
    assert result.error_count == 1


def test_optimize_embedding_call_caches_result(tmp_path):
    engine = make_engine(tmp_path)
    calls = []

    def embed(text):
        calls.append(text)
        return [len(text)]

    engine.optimize_embedding_call("hello world", embed)
    engine.optimize_embedding_call("hello world", embed)
    assert len(calls) == 1


def test_optimize_batch_embedding_only_computes_misses(tmp_path):
    engine = make_engine(tmp_path)
    engine.optimize_embedding_call("already cached", lambda t: [999])

    batch_calls = []

    def embed_batch(texts):
        batch_calls.append(texts)
        return [[len(t)] for t in texts]

    results = engine.optimize_batch_embedding(
        ["already cached", "new text one", "new text two"], embed_batch
    )
    assert results[0] == [999]
    assert len(batch_calls) == 1
    assert batch_calls[0] == ["new text one", "new text two"]


def test_process_in_batches_streams_all_items(tmp_path):
    engine = make_engine(tmp_path)

    def batch_fn(batch):
        return [x * 2 for x in batch]

    results = engine.process_in_batches(range(10), batch_fn, batch_size=3)
    assert sorted(results) == [x * 2 for x in range(10)]


def test_refine_entities_delegates_correctly(tmp_path):
    engine = make_engine(tmp_path)
    result = engine.refine_entities(["Python", "python"])
    assert result.output_count == 1


def test_repair_noisy_text_delegates_correctly(tmp_path):
    engine = make_engine(tmp_path)
    cleaned, report = engine.repair_noisy_text("Header----------\nn\nReal content here")
    assert "----------" not in cleaned
    assert report.orphan_lines_removed == 1


def test_run_benchmark_and_save_writes_report(tmp_path):
    engine = make_engine(tmp_path)
    report = engine.run_benchmark_and_save()

    assert "metadata" in report
    assert "benchmark" in report
    assert "improvement_backlog" in report
    assert (tmp_path / "performance_report.json").exists()


def test_run_benchmark_reports_pipeline_mode_honestly(tmp_path):
    """
    Regression guard: the benchmark must accurately report whether the
    upstream Day 5-17 pipeline is importable and must provide a matching
    explanatory note.
    """
    engine = make_engine(tmp_path)
    report = engine.run_benchmark_and_save()

    benchmark = report["benchmark"]

    assert isinstance(benchmark["upstream_pipeline_importable"], bool)
    assert isinstance(benchmark["fallback_note"], str)

    if benchmark["upstream_pipeline_importable"]:
        assert "live pipeline" in benchmark["fallback_note"].lower()
    else:
        assert "not importable" in benchmark["fallback_note"].lower()
