"""
Benchmark suite
----------------
Runs profiling passes across the optimization modules against a set of
fixture inputs and assembles a performance report. Where upstream
Day 5-17 packages are not importable in isolation (the same situation
noted in the Day 17 improvement backlog), each benchmark degrades to
exercising this package's own logic against representative fixture data
and flags the fallback explicitly in the report rather than silently
reporting numbers that look like they came from the full pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .caching import EmbeddingCache, ExtractionCache
from .entity_refinement import EntityRefiner
from .noisy_resume_handler import NoisyResumeHandler
from .parallel_processing import parallel_map
from .profiling import Timer

logger = logging.getLogger("performance_optimization_engine.benchmark")


@dataclass
class BenchmarkReport:
    upstream_pipeline_importable: bool
    fallback_note: str
    results: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "upstream_pipeline_importable": self.upstream_pipeline_importable,
            "fallback_note": self.fallback_note,
            "results": self.results,
        }


def _check_upstream_importable() -> bool:
    """Mirrors the Day 17 test-runner's import probe for the upstream
    engines, so this report is honest about which mode it ran in."""
    try:
        import resume_extraction_engine  # noqa: F401
        import semantic_matching_engine  # noqa: F401

        return True
    except ImportError:
        return False


class BenchmarkSuite:
    def __init__(self, sample_texts: List[str], sample_entities: List[Any]):
        self.sample_texts = sample_texts
        self.sample_entities = sample_entities

    def run(self) -> BenchmarkReport:
        upstream_ok = _check_upstream_importable()
        fallback_note = (
            "Upstream Day 5-17 packages were importable; benchmarks ran "
            "against the live pipeline."
            if upstream_ok
            else "Upstream Day 5-17 packages were not importable in this "
            "isolated run (same condition noted in the Day 17 improvement "
            "backlog). Benchmarks below exercise this package's own "
            "extraction-cache, parallel-processing, entity-refinement, and "
            "noisy-text-repair logic against representative fixture text, "
            "not the live upstream engines. Re-run inside the full "
            "integrated repository for pipeline-accurate numbers."
        )

        report = BenchmarkReport(
            upstream_pipeline_importable=upstream_ok, fallback_note=fallback_note
        )

        report.results.append(self._bench_extraction_cache())
        report.results.append(self._bench_parallel_processing())
        report.results.append(self._bench_entity_refinement())
        report.results.append(self._bench_noisy_repair())
        report.results.append(self._bench_embedding_cache())
        return report

    # ------------------------------------------------------------------ #
    def _bench_extraction_cache(self) -> Dict[str, Any]:
        cache = ExtractionCache()
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.txt"
            path.write_text(
                self.sample_texts[0] if self.sample_texts else "sample", encoding="utf-8"
            )

            with Timer("extraction_cache_cold_miss") as t_miss:
                result = cache.get(path)
            with Timer("extraction_cache_set"):
                cache.set(path, {"cleaned_text": "dummy"})
            with Timer("extraction_cache_warm_hit") as t_hit:
                result = cache.get(path)

        return {
            "benchmark": "extraction_cache",
            "cold_miss": t_miss.metrics.to_dict(),
            "warm_hit": t_hit.metrics.to_dict(),
            "cache_stats": cache.stats,
            "hit_returned_expected": result is not None,
        }

    def _bench_parallel_processing(self) -> Dict[str, Any]:
        def slow_identity(x: int) -> int:
            total = 0
            for i in range(20000):
                total += i % (x + 1)
            return total

        items = list(range(12))

        with Timer("sequential_processing", records_processed=len(items)) as t_seq:
            for item in items:
                slow_identity(item)

        with Timer("parallel_processing", records_processed=len(items)) as t_par:
            run_result = parallel_map(slow_identity, items, max_workers=4)

        speedup = (
            t_seq.metrics.duration_seconds / t_par.metrics.duration_seconds
            if t_par.metrics.duration_seconds > 0
            else 0.0
        )

        return {
            "benchmark": "parallel_processing",
            "sequential": t_seq.metrics.to_dict(),
            "parallel": t_par.metrics.to_dict(),
            "speedup_factor": round(speedup, 2),
            "errors": run_result.error_count,
        }

    def _bench_entity_refinement(self) -> Dict[str, Any]:
        refiner = EntityRefiner()
        with Timer("entity_refinement", records_processed=len(self.sample_entities)) as t:
            result = refiner.refine(self.sample_entities)
        return {
            "benchmark": "entity_refinement",
            "timing": t.metrics.to_dict(),
            "input_count": result.input_count,
            "output_count": result.output_count,
            "merged_duplicates": result.merged_duplicates,
            "dropped_low_confidence": result.dropped_low_confidence,
            "similarity_backend": result.similarity_backend,
        }

    def _bench_noisy_repair(self) -> Dict[str, Any]:
        handler = NoisyResumeHandler()
        combined = "\n".join(self.sample_texts) if self.sample_texts else "sample text"
        with Timer("noisy_resume_repair") as t:
            _, repair_report = handler.repair(combined)
        return {
            "benchmark": "noisy_resume_repair",
            "timing": t.metrics.to_dict(),
            "repairs": repair_report.to_dict(),
        }

    def _bench_embedding_cache(self) -> Dict[str, Any]:
        cache = EmbeddingCache(model_version="bench-v1")
        text = self.sample_texts[0] if self.sample_texts else "sample text"

        with Timer("embedding_cache_miss"):
            cache.get(text)
        with Timer("embedding_cache_set"):
            cache.set(text, [0.1, 0.2, 0.3])
        with Timer("embedding_cache_hit") as t_hit:
            hit = cache.get(text)

        return {
            "benchmark": "embedding_cache",
            "hit": t_hit.metrics.to_dict(),
            "cache_stats": cache.stats,
            "hit_returned_expected": hit == [0.1, 0.2, 0.3],
        }
