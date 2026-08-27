"""
storage.py
----------
Persists ScoringResult objects as structured JSON carrying the Day 7
metadata envelope (schema_version, model_version, pipeline_version,
candidate_id, job_id), plus the full explainable breakdown.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .explainability import build_component_explanations, build_narrative
from .scoring_engine import ScoringResult

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "ats-scoring-engine-1.0.0"
PIPELINE_VERSION = "zecpath-day13"


@dataclass
class ScoreRecord:
    schema_version: str
    model_version: str
    pipeline_version: str
    candidate_id: str
    job_id: str
    generated_at: str
    role_profile: str
    status: str
    final_score: Optional[float]
    missing_data_mode: str
    components_missing: List[str]
    weights_renormalized: bool
    components: List[Dict[str, Any]]
    narrative_explanation: str
    component_explanations: Dict[str, str]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_score_record(result: ScoringResult) -> ScoreRecord:
    """Wrap a ScoringResult with the metadata envelope + explainability text."""
    return ScoreRecord(
        schema_version=SCHEMA_VERSION,
        model_version=MODEL_VERSION,
        pipeline_version=PIPELINE_VERSION,
        candidate_id=result.candidate_id,
        job_id=result.job_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        role_profile=result.role_profile,
        status=result.status,
        final_score=result.final_score,
        missing_data_mode=result.missing_data_mode,
        components_missing=result.components_missing,
        weights_renormalized=result.weights_renormalized,
        components=[
            {
                "name": c.name,
                "available": c.available,
                "raw_score": c.raw_score,
                "weight_original": c.weight_original,
                "weight_applied": c.weight_applied,
                "weighted_contribution": c.weighted_contribution,
                "notes": c.notes,
            }
            for c in result.components
        ],
        narrative_explanation=build_narrative(result),
        component_explanations=build_component_explanations(result),
        warnings=result.warnings,
    )


class ResultStore:
    """Writes ScoreRecord objects to disk as structured JSON."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, record: ScoreRecord) -> str:
        safe_candidate = str(record.candidate_id).replace("/", "_").replace("\\", "_")
        safe_job = str(record.job_id).replace("/", "_").replace("\\", "_")
        out_path = self.output_dir / f"{safe_candidate}__{safe_job}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)
        return str(out_path)

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
