# Performance Optimization & Tuning Engine

**Day 18 deliverable — Zecpath AI Job Portal**

Makes the ATS pipeline production-ready by targeting the five areas in
the Day 18 brief: text extraction speed, model response time, memory
handling, entity detection accuracy, and noisy-resume handling. Every
piece is additive — this package never imports or modifies Day 5-17
modules. It wraps arbitrary callables (extraction functions, embedding
functions) and post-processes arbitrary structured output (entity
lists, cleaned text) via the same alias-tolerant pattern used across
the rest of the repository.

## What it does

1. **Optimizes text extraction speed** — `ExtractionCache` skips
   re-extracting unchanged files (keyed on content hash + mtime), and
   `parallel_processing.BatchExtractor` fans genuine cache misses out
   across a thread pool (threads, not processes, since extraction is
   I/O-bound and PDF/DOCX C-extensions release the GIL).
2. **Reduces model response time** — `EmbeddingCache` caches embedding
   vectors keyed on model version + normalized text, so repeated
   semantic-matching calls for the same resume/JD pair skip the
   sentence-transformers forward pass entirely; `optimize_batch_embedding`
   batches only genuine cache misses into a single model call.
3. **Improves memory handling** — `chunked` / `iter_text_pages` stream
   large batches and long documents instead of materializing them
   wholesale, and `MemoryBudgetGuard` fails a batch job fast (or just
   warns) when RSS crosses a configured budget, instead of letting it
   OOM silently in production.
4. **Refines entity detection** — `EntityRefiner` deduplicates near-
   duplicate skills/entities ("Node.js" / "NodeJS" / "node js"),
   merges and boosts confidence for entities confirmed by multiple
   extraction passes, drops low-confidence noise, and canonicalizes
   casing while preserving real acronyms (AWS, SQL).
5. **Improves noisy resume handling** — `NoisyResumeHandler` adds a
   repair pass on top of (never replacing) the Day 5 `TextCleaner`:
   collapsing repeated punctuation runs, dropping orphaned single-
   character line fragments (the "n" artifact seen in the Day 5 noisy
   PDF fixture), OCR character-confusion repair, and corpus-level
   boilerplate/watermark removal across a batch of resumes.

## Project layout

```
performance_optimization_engine/
├── __init__.py
├── optimizer.py              # PerformanceOptimizationEngine — main orchestrator
├── profiling.py                # Timer / timed() — duration + memory instrumentation
├── caching.py                   # LRUCache, ExtractionCache, EmbeddingCache
├── parallel_processing.py        # parallel_map, BatchExtractor
├── memory_optimizer.py            # chunked, iter_text_pages, MemoryBudgetGuard
├── entity_refinement.py            # EntityRefiner — dedup/merge/canonicalize
├── noisy_resume_handler.py          # NoisyResumeHandler — extra noise repair
├── benchmark.py                      # BenchmarkSuite — assembles the performance report
├── storage.py                          # PerformanceReport + Day 7 metadata envelope
├── cli.py                                # command-line entry point
└── tests/
    ├── test_*.py                          # pytest suite (70 tests)
    ├── run_tests.py                        # runs the suite + writes a timestamped log
    ├── logs/                                # generated test-run logs (deliverable)
    └── outputs/                              # generated performance_report.json
```

## Requirements

```
pip install pytest rapidfuzz psutil --break-system-packages
```

Every optional accelerator (`rapidfuzz`, `psutil`, `wordfreq`) degrades
gracefully to a slower stdlib fallback — with a logged warning — when
absent, following the same two-pass pattern established in Day 12-15.

## Usage

### Python API

```python
from performance_optimization_engine import PerformanceOptimizationEngine

engine = PerformanceOptimizationEngine(output_dir="outputs", max_workers=4)

# 1. Extraction speed — wraps any single-file extractor
result = engine.optimize_extraction(my_extract_fn, resume_paths)

# 2. Model response time — wraps any embedding function
vector = engine.optimize_embedding_call(resume_text, my_embed_fn)
vectors = engine.optimize_batch_embedding(texts, my_embed_batch_fn)

# 3. Memory handling — streams a large candidate pool in bounded batches
scored = engine.process_in_batches(candidates, my_scoring_batch_fn, batch_size=25)

# 4. Entity refinement
refined = engine.refine_entities(raw_skill_list_or_dict)

# 5. Noisy resume repair
repaired_text, report = engine.repair_noisy_text(cleaned_text)
deduped_docs, report = engine.deboilerplate_batch(list_of_cleaned_resumes)

# Full benchmark + report
report = engine.run_benchmark_and_save()
```

### CLI

```
python -m performance_optimization_engine.cli benchmark --output-dir outputs
python -m performance_optimization_engine.cli refine-entities skills.json
python -m performance_optimization_engine.cli clean-noisy resume.clean.txt
```

## Running tests

```
python performance_optimization_engine/tests/run_tests.py
```

70 tests covering: timer/memory profiling accuracy, LRU/extraction/
embedding cache correctness (including alias tolerance and cache
invalidation on file content change), parallel-map correctness and
per-item error isolation, memory chunking/paging boundary conditions,
entity-refinement dedup/confidence/casing rules and malformed-input
handling, noisy-text repair and corpus boilerplate detection, metadata
envelope compliance, and end-to-end orchestrator integration.

## Known limitations & honest benchmarking

- **Benchmark fixtures, not live pipeline**: Day 5-17 packages
  (`resume_extraction_engine`, `semantic_matching_engine`, etc.) are not
  importable in this isolated package — the same condition flagged in
  the Day 17 improvement backlog. `BenchmarkSuite` detects this at
  runtime and labels its report `upstream_pipeline_importable: false`
  with an explicit fallback note, rather than presenting fixture-based
  numbers as if they were measured against the live pipeline. Re-run
  `benchmark` inside the fully integrated repository for pipeline-
  accurate throughput numbers (tracked as backlog item PERF-001).
- **Thread-pool speedup is workload-dependent**: `parallel_map` helps
  most for I/O-bound work (file reads, PDF/DOCX parsing) on
  multi-core hardware. On a single-core CI runner, or for pure
  CPU-bound Python loops that hold the GIL, threading shows little or
  no speedup — this is expected and is reported honestly rather than
  hidden.
- **Optional accelerators**: `rapidfuzz`, `psutil`, and `wordfreq` are
  not hard dependencies. Without them, entity dedup, memory
  measurement, and OCR-repair dictionary confirmation all still run,
  just with slower/less-precise fallbacks (tracked as PERF-002).
