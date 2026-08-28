"""
storage.py
----------
Persists FairnessProcessingResult objects as structured JSON, following
the Day 7 metadata envelope convention (schema_version, model_version,
pipeline_version) applied consistently across all storage outputs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "fairness-bias-engine-1.0"
PIPELINE_VERSION = "day15"


@dataclass
class FairnessProcessingRecord:
    generated_at: str
    role_id: Optional[str]
    candidate_count: int
    masked_profiles: List[Dict[str, Any]]
    keyword_balance_reports: Dict[str, Dict[str, Any]]
    normalized_scores: List[Dict[str, Any]]
    bias_reports: List[Dict[str, Any]]
    schema_version: str = SCHEMA_VERSION
    model_version: str = MODEL_VERSION
    pipeline_version: str = PIPELINE_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FairnessResultStore:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, record: FairnessProcessingRecord, filename: str) -> str:
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)
        return str(path)

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
