"""
Structured Storage
-------------------
Serializes ExperienceParser + relevance-scoring output into the
metadata-envelope-compatible record shapes used across the Zecpath
pipeline (candidate_id, schema_version, etc. — see Day 7 design doc),
and persists them as JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .gaps import ExperienceTimeline
from .parser import ExperienceParseResult
from .relevance import ExperienceRelevanceResult

SCHEMA_VERSION = "1.0.0"


@dataclass
class ExperienceRecord:
    """Structured experience object — the Day 10 core deliverable."""

    candidate_id: str
    source_file: str
    schema_version: str
    extracted_at: str
    entries: List[Dict[str, Any]]
    total_experience_months: int
    total_experience_years: float
    valid_entry_count: int
    skipped_entry_count: int
    gaps: List[Dict[str, Any]]
    overlaps: List[Dict[str, Any]]
    section_found: bool
    unmatched_line_count: int
    warnings: List[str] = field(default_factory=list)
    status: str = "success"  # "success" | "partial" | "failed"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperienceRelevanceRecord:
    """Wraps ExperienceRelevanceResult with pipeline metadata envelope fields."""

    candidate_id: str
    job_id: str
    schema_version: str
    scored_at: str
    target_title: str
    overall_relevance_score: float
    relevant_experience_years: float
    total_experience_years: float
    role_scores: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_experience_record(
    source_file: str,
    candidate_id: str,
    parse_result: ExperienceParseResult,
    timeline: ExperienceTimeline,
) -> ExperienceRecord:
    warnings: List[str] = []
    if not parse_result.section_found:
        warnings.append(
            "No canonical 'Experience' section heading was found; the whole "
            "document was scanned for role-header lines."
        )
    if parse_result.lines_unmatched_in_section:
        warnings.append(
            f"{len(parse_result.lines_unmatched_in_section)} line(s) in the experience "
            "section did not match the expected 'Title — Company (Start – End)' pattern."
        )
    if timeline.skipped_entry_count:
        warnings.append(
            f"{timeline.skipped_entry_count} entry(ies) had unparseable dates and were "
            "excluded from total-experience/gap/overlap calculations."
        )

    if not parse_result.entries:
        status = "failed"
        warnings.append("No experience entries could be extracted from this document.")
    elif warnings:
        status = "partial"
    else:
        status = "success"

    return ExperienceRecord(
        candidate_id=candidate_id,
        source_file=source_file,
        schema_version=SCHEMA_VERSION,
        extracted_at=now_iso(),
        entries=[asdict(e) for e in parse_result.entries],
        total_experience_months=timeline.total_months,
        total_experience_years=timeline.total_years,
        valid_entry_count=timeline.valid_entry_count,
        skipped_entry_count=timeline.skipped_entry_count,
        gaps=[asdict(g) for g in timeline.gaps],
        overlaps=[asdict(o) for o in timeline.overlaps],
        section_found=parse_result.section_found,
        unmatched_line_count=len(parse_result.lines_unmatched_in_section),
        warnings=warnings,
        status=status,
    )


def build_relevance_record(
    candidate_id: str, job_id: str, result: ExperienceRelevanceResult
) -> ExperienceRelevanceRecord:
    return ExperienceRelevanceRecord(
        candidate_id=candidate_id,
        job_id=job_id,
        schema_version=SCHEMA_VERSION,
        scored_at=now_iso(),
        target_title=result.target_title,
        overall_relevance_score=result.overall_relevance_score,
        relevant_experience_years=result.relevant_experience_years,
        total_experience_years=result.total_experience_years,
        role_scores=[asdict(rs) for rs in result.role_scores],
    )


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ResultStore:
    """Writes ExperienceRecord / ExperienceRelevanceRecord objects to disk as JSON."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.experience_dir = self.output_dir / "experience"
        self.relevance_dir = self.output_dir / "relevance"
        self.experience_dir.mkdir(parents=True, exist_ok=True)
        self.relevance_dir.mkdir(parents=True, exist_ok=True)

    def save_experience(self, record: ExperienceRecord) -> str:
        stem = Path(record.source_file).stem or record.candidate_id
        path = self.experience_dir / f"{stem}.experience.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)
        return str(path)

    def save_relevance(self, record: ExperienceRelevanceRecord) -> str:
        path = self.relevance_dir / f"{record.candidate_id}__{record.job_id}.relevance.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)
        return str(path)
