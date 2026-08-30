"""
Performance Optimization Engine
---------------------------------
Top-level orchestrator combining caching, parallel processing, memory
guarding, entity refinement, noisy-text repair, and benchmarking into a
single entry point — the Day 18 equivalent of `ResumeExtractionEngine`
or `ATSScoringEngine` from prior days.

Additive-only: never imports or mutates Day 5-17 modules directly.
Callers pass in their own extraction/scoring callables and data; this
engine wraps them with optimization behavior.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from .benchmark import BenchmarkSuite
from .caching import EmbeddingCache, ExtractionCache
from .entity_refinement import EntityRefiner, RefinementResult
from .memory_optimizer import MemoryBudgetGuard, chunked
from .noisy_resume_handler import NoiseRepairReport, NoisyResumeHandler
from .parallel_processing import BatchExtractor, ParallelRunResult
from .storage import PerformanceReport, ReportStore, build_metadata_envelope

logger = logging.getLogger("performance_optimization_engine")


class PerformanceOptimizationEngine:
    """
    Orchestrates all Day 18 optimizations. Each capability is exposed as
    an independent method so callers can adopt them incrementally
    (e.g. wrap just extraction, or just entity refinement) without
    committing to the whole engine.
    """

    def __init__(
        self,
        output_dir: str | Path = "outputs",
        max_workers: int = 4,
        memory_budget_mb: Optional[float] = None,
        embedding_model_version: str = "unknown",
    ):
        self.extraction_cache = ExtractionCache()
        self.embedding_cache = EmbeddingCache(model_version=embedding_model_version)
        self.entity_refiner = EntityRefiner()
        self.noisy_handler = NoisyResumeHandler()
        self.max_workers = max_workers
        self.memory_budget_mb = memory_budget_mb
        self.store = ReportStore(output_dir)

    # ------------------------------------------------------------------ #
    # 1. Optimize text extraction speed
    # ------------------------------------------------------------------ #
    def optimize_extraction(
        self, process_file_fn: Callable[[Any], Any], file_paths: Iterable[Any]
    ) -> ParallelRunResult:
        """
        Wraps an arbitrary single-file extraction callable with a
        cache-then-parallelize strategy: cache hits are resolved
        immediately (sequentially, since they're just a dict lookup),
        and only genuine cache misses are dispatched to the thread pool.
        """
        file_list = list(file_paths)
        cached_results: List[Any] = []
        misses: List[Any] = []

        for path in file_list:
            hit = self.extraction_cache.get(path)
            if hit is not None:
                cached_results.append(hit)
            else:
                misses.append(path)

        def _process_and_cache(path: Any) -> Any:
            record = process_file_fn(path)
            self.extraction_cache.set(path, record)
            return record

        batch = BatchExtractor(_process_and_cache, max_workers=self.max_workers)
        run_result = batch.process_files(misses)
        run_result.results = cached_results + run_result.results
        run_result.total_items = len(file_list)

        logger.info(
            "optimize_extraction: %d total, %d cache hits, %d processed, %d errors",
            len(file_list),
            len(cached_results),
            len(run_result.results) - len(cached_results),
            run_result.error_count,
        )
        return run_result

    # ------------------------------------------------------------------ #
    # 2. Reduce model response time
    # ------------------------------------------------------------------ #
    def optimize_embedding_call(self, text: Any, embed_fn: Callable[[str], Any]) -> Any:
        """
        Wraps an embedding-model call with the embedding cache so
        repeated calls for identical (or whitespace/case-normalized
        identical) text never re-invoke the model.
        """
        cached = self.embedding_cache.get(text)
        if cached is not None:
            return cached
        embedding = embed_fn(self.embedding_cache._extract_text(text))
        self.embedding_cache.set(text, embedding)
        return embedding

    def optimize_batch_embedding(
        self, texts: List[Any], embed_batch_fn: Callable[[List[str]], List[Any]]
    ) -> List[Any]:
        """
        Batches only the cache misses into a single call to the
        (comparatively expensive) batch embedding function, then
        reassembles results in original order.
        """
        results: List[Any] = [None] * len(texts)
        miss_indices: List[int] = []
        miss_texts: List[str] = []

        for i, text in enumerate(texts):
            cached = self.embedding_cache.get(text)
            if cached is not None:
                results[i] = cached
            else:
                miss_indices.append(i)
                miss_texts.append(self.embedding_cache._extract_text(text))

        if miss_texts:
            computed = embed_batch_fn(miss_texts)
            for text_idx, embedding in zip(miss_indices, computed):
                results[text_idx] = embedding
                self.embedding_cache.set(texts[text_idx], embedding)

        return results

    # ------------------------------------------------------------------ #
    # 3. Improve memory handling
    # ------------------------------------------------------------------ #
    def process_in_batches(
        self,
        items: Iterable[Any],
        batch_fn: Callable[[List[Any]], List[Any]],
        batch_size: int = 25,
    ) -> List[Any]:
        """
        Streams `items` through `batch_fn` in bounded-size chunks under a
        MemoryBudgetGuard, so a large candidate pool never needs to be
        fully materialized (input + all intermediate results) at once.
        """
        results: List[Any] = []
        with MemoryBudgetGuard(budget_mb=self.memory_budget_mb) as guard:
            for batch in chunked(items, batch_size):
                results.extend(batch_fn(batch))
                guard.checkpoint()
        if guard.peak_mb is not None:
            logger.info("process_in_batches peak RSS: %.1f MB", guard.peak_mb)
        return results

    # ------------------------------------------------------------------ #
    # 4. Refine entity detection
    # ------------------------------------------------------------------ #
    def refine_entities(self, entity_source: Any) -> RefinementResult:
        return self.entity_refiner.refine(entity_source)

    # ------------------------------------------------------------------ #
    # 5. Improve noisy resume handling
    # ------------------------------------------------------------------ #
    def repair_noisy_text(self, cleaned_text: str) -> tuple[str, NoiseRepairReport]:
        return self.noisy_handler.repair(cleaned_text)

    def deboilerplate_batch(self, documents: List[str]) -> tuple[List[str], NoiseRepairReport]:
        return self.noisy_handler.remove_corpus_boilerplate(documents)

    # ------------------------------------------------------------------ #
    # Benchmark + report
    # ------------------------------------------------------------------ #
    def run_benchmark_and_save(
        self, sample_texts: Optional[List[str]] = None, sample_entities: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        suite = BenchmarkSuite(
            sample_texts=sample_texts or ["Sample resume text for benchmarking purposes."],
            sample_entities=sample_entities
            or ["Python", "python", "PYTHON", "AWS", "Amazon Web Services", "SQL"],
        )
        benchmark_report = suite.run()

        report = PerformanceReport(
            metadata=build_metadata_envelope(model_version="day18-benchmark-1.0.0"),
            benchmark=benchmark_report.to_dict(),
            improvement_backlog=[
                {
                    "id": "PERF-001",
                    "description": (
                        "Re-run this benchmark suite inside the fully integrated "
                        "repository once Day 5-17 packages are importable in "
                        "isolation, mirroring the same open item from the Day 17 "
                        "backlog, to get pipeline-accurate (not fixture-based) "
                        "throughput numbers."
                    ),
                    "status": (
                        "open" if not benchmark_report.upstream_pipeline_importable else "resolved"
                    ),
                },
                {
                    "id": "PERF-002",
                    "description": (
                        "Install optional accelerators (rapidfuzz, wordfreq, "
                        "psutil) in the production environment; several "
                        "optimizations here silently degrade to slower stdlib "
                        "fallbacks without them."
                    ),
                    "status": "open",
                },
            ],
        )
        path = self.store.save(report)
        logger.info("Performance report written to %s", path)
        return report.to_dict()
