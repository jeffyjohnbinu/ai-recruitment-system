"""
Structured Storage
-------------------
Data models for the Candidate Ranking & Shortlisting Engine, plus
persistence of ranked results as structured JSON (for downstream/API
consumption) and CSV (for recruiters to open in Excel/Sheets).

Follows the Day 7 metadata envelope convention: every output record
carries candidate_id, job_id, schema_version, model_version, and
pipeline_version fields.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.0"
MODEL_VERSION = "candidate_ranking_engine-1.0"
PIPELINE_VERSION = "zecpath-pipeline-day14"


@dataclass
class CandidateMatchInput:
    """
    Normalized view of a single candidate/job match record, as produced
    by the Day 12 Semantic Matching Engine (or any upstream stage that
    emits a comparable envelope). Loader tolerates minor key variations
    so this module never requires changes to Day 12's output format.
    """

    candidate_id: str
    job_id: str
    overall_score: float
    band: Optional[str] = None
    section_scores: Dict[str, float] = field(default_factory=dict)
    candidate_name: Optional[str] = None
    source_file: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RankedCandidate:
    rank: int
    candidate_id: str
    job_id: str
    candidate_name: Optional[str]
    overall_score: float
    band: Optional[str]
    zone: str  # "Shortlist" | "Review" | "Auto-Reject"
    section_scores: Dict[str, float]
    reason: str
    schema_version: str = SCHEMA_VERSION
    model_version: str = MODEL_VERSION
    pipeline_version: str = PIPELINE_VERSION
    generated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RankingReport:
    job_id: str
    generated_at: str
    total_candidates: int
    shortlist_count: int
    review_count: int
    auto_reject_count: int
    shortlist_threshold: float
    review_threshold: float
    top_n: int
    top_candidates: List[RankedCandidate]
    ranked_candidates: List[RankedCandidate]
    schema_version: str = SCHEMA_VERSION
    model_version: str = MODEL_VERSION
    pipeline_version: str = PIPELINE_VERSION

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


class ResultStore:
    """Writes RankingReport objects to disk as JSON + a recruiter-friendly CSV."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.json_dir = self.output_dir / "structured"
        self.csv_dir = self.output_dir / "recruiter_csv"
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir.mkdir(parents=True, exist_ok=True)

    def save(self, report: RankingReport) -> Dict[str, str]:
        safe_job_id = _safe_filename(report.job_id)
        json_path = self.json_dir / f"{safe_job_id}_ranking.json"
        csv_path = self.csv_dir / f"{safe_job_id}_ranking.csv"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

        self._write_csv(report, csv_path)

        return {"json": str(json_path), "csv": str(csv_path)}

    @staticmethod
    def _write_csv(report: RankingReport, csv_path: Path) -> None:
        fieldnames = [
            "rank",
            "candidate_id",
            "candidate_name",
            "job_id",
            "overall_score",
            "band",
            "zone",
            "reason",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for c in report.ranked_candidates:
                writer.writerow(
                    {
                        "rank": c.rank,
                        "candidate_id": c.candidate_id,
                        "candidate_name": c.candidate_name or "",
                        "job_id": c.job_id,
                        "overall_score": f"{c.overall_score:.4f}",
                        "band": c.band or "",
                        "zone": c.zone,
                        "reason": c.reason,
                    }
                )

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()


def _safe_filename(value: str) -> str:
    keep = "-_." + "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(c if c in keep else "_" for c in value) or "unknown_job"
