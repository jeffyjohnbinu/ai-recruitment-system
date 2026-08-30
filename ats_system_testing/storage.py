"""
storage.py
-----------
Persists a Day 17 test run as structured JSON, wrapped in the Day 7
metadata envelope standard (schema_version, model_version,
pipeline_version, request_id, generated_at) so downstream tooling
(dashboards, CI artifacts) can consume it the same way every other
day's output is consumed.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from .harness import CaseResult
from .metrics import MetricsReport

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "day17-ats-testing-1.0.0"
PIPELINE_VERSION = "zecpath-ats-pipeline-day16"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_envelope(results: List[CaseResult], metrics: MetricsReport) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "request_id": str(uuid.uuid4()),
        "generated_at": _now_iso(),
        "case_count": len(results),
        "cases": [
            {
                "case_id": r.test_case.case_id,
                "role_type": r.test_case.role_type,
                "seniority": r.test_case.seniority,
                "expected_outcome": r.test_case.expected_outcome,
                "candidate_name": r.test_case.candidate_name,
                "tags": r.test_case.tags,
                "ground_truth": asdict(r.test_case.ground_truth),
                "ai_output": {
                    "score": r.pipeline_output.score,
                    "shortlisted": r.pipeline_output.shortlisted,
                    "recommendation": r.pipeline_output.recommendation,
                    "matched_keywords": r.pipeline_output.matched_keywords,
                    "engine_used": r.pipeline_output.engine_used,
                    "warnings": r.pipeline_output.warnings,
                },
                "is_correct": r.is_correct,
            }
            for r in results
        ],
        "metrics": metrics.to_dict(),
    }


class ResultStore:
    """Writes a Day 17 test run (envelope + metrics) to disk as JSON."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, results: List[CaseResult], metrics: MetricsReport) -> Path:
        envelope = build_envelope(results, metrics)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"ats_test_run_{timestamp}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(envelope, f, indent=2, ensure_ascii=False)
        return path

    @staticmethod
    def now_iso() -> str:
        return _now_iso()
