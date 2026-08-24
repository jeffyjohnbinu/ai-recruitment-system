"""
Structured Storage
-------------------
Persists SemanticMatchRecord objects as JSON, following the same metadata
envelope convention (candidate_id, job_id, schema_version) established in
the Day 7 pipeline & storage design, so downstream stages (ranking,
shortlisting, reporting) can consume this module's output the same way
they consume every other day's.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

SCHEMA_VERSION = "1.0.0"


@dataclass
class SemanticMatchRecord:
    candidate_id: str
    job_id: str
    schema_version: str
    matched_at: str
    engine_used: str
    overall_score: float
    match_band: str  # "strong_match" | "possible_match" | "no_match"
    is_match: bool
    section_scores: Dict[str, Any] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    status: str = "success"  # "success" | "partial" | "failed"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResultStore:
    """Writes SemanticMatchRecord objects to disk as JSON."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.json_dir = self.output_dir / "structured" / "semantic_matches"
        self.json_dir.mkdir(parents=True, exist_ok=True)

    def save(self, record: SemanticMatchRecord) -> str:
        filename = f"{record.candidate_id}__{record.job_id}.json"
        json_path = self.json_dir / filename
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)
        return str(json_path)

    def save_report(
        self, report: Dict[str, Any], filename: str = "matching_accuracy_report.json"
    ) -> str:
        report_path = self.output_dir / "structured" / "semantic_matches" / filename
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return str(report_path)

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
