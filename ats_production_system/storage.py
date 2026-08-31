"""
storage.py
-----------
Day 20 deliverable — Zecpath AI Job Portal

Persists one end-to-end ATSPipeline run (one candidate scored against one
job) as structured JSON carrying the Day 7 metadata envelope
(schema_version, model_version, pipeline_version, candidate_id, job_id,
generated_at, request_id) established in Day 7 and used by every module
since.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "ats-production-system-1.0.0"
PIPELINE_VERSION = "zecpath-day20"


@dataclass
class PipelineRunRecord:
    schema_version: str
    model_version: str
    pipeline_version: str
    candidate_id: str
    job_id: str
    generated_at: str
    request_id: str

    resume_source_file: str
    job_source_file: str

    stages_completed: List[str]
    stages_failed: List[str]
    engines_used: Dict[str, str]

    final_score: Optional[float]
    match_band: Optional[str]
    recommendation: Optional[str]
    zone: Optional[str]

    component_breakdown: List[Dict[str, Any]]
    narrative_explanation: Optional[str]
    warnings: List[str]

    status: str  # "success" | "partial" | "failed"
    error: Optional[str] = None
    candidate_profile: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BatchRunSummary:
    schema_version: str
    generated_at: str
    job_id: str
    total_candidates: int
    shortlist_count: int
    review_count: int
    auto_reject_count: int
    candidates: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PipelineResultStore:
    """Writes PipelineRunRecord / BatchRunSummary objects to disk as JSON."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.runs_dir = self.output_dir / "structured" / "pipeline_runs"
        self.batches_dir = self.output_dir / "structured" / "batch_runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.batches_dir.mkdir(parents=True, exist_ok=True)

    def save_run(self, record: PipelineRunRecord) -> str:
        filename = f"{record.candidate_id}__{record.job_id}.json"
        path = self.runs_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)
        return str(path)

    def save_batch(self, summary: BatchRunSummary) -> str:
        filename = f"batch__{summary.job_id}.json"
        path = self.batches_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False)
        return str(path)

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def new_request_id() -> str:
        return str(uuid.uuid4())
