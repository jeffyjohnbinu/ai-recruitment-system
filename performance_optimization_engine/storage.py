"""
Structured storage
-------------------
Persists performance reports as JSON, using the Day 7 metadata envelope
standard (`schema_version`, `model_version`, `pipeline_version`,
`candidate_id`, `job_id`, `generated_at`, `request_id`) so this module's
output is consumable by the same downstream tooling as every other
day's structured JSON.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

SCHEMA_VERSION = "1.0.0"
PIPELINE_VERSION = "day18-performance-optimization-1.0.0"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_request_id() -> str:
    return str(uuid.uuid4())


def build_metadata_envelope(
    candidate_id: Optional[str] = None,
    job_id: Optional[str] = None,
    model_version: str = "n/a",
) -> Dict[str, Any]:
    """Day 7 standard envelope, reused verbatim across every module."""
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": model_version,
        "pipeline_version": PIPELINE_VERSION,
        "candidate_id": candidate_id,
        "job_id": job_id,
        "generated_at": now_iso(),
        "request_id": new_request_id(),
    }


@dataclass
class PerformanceReport:
    metadata: Dict[str, Any]
    benchmark: Dict[str, Any]
    improvement_backlog: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata,
            "benchmark": self.benchmark,
            "improvement_backlog": self.improvement_backlog,
        }


class ReportStore:
    """Writes PerformanceReport objects to disk as JSON."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, report: PerformanceReport, filename: str = "performance_report.json") -> str:
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        return str(path)
