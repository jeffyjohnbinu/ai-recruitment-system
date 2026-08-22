"""
storage.py
----------
Persists the structured academic profile (degrees + certifications) as
JSON, following the same output-directory / naming conventions as Day 5's
ResultStore (`outputs/structured/<name>.json`), so downstream tooling and
CI artifact uploads can treat every day's output directory the same way.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .certification_parser import CertificationRecord
from .degree_parser import DegreeRecord

SCHEMA_VERSION = "1.0.0"


@dataclass
class AcademicProfileRecord:
    source_file: str
    schema_version: str
    extracted_at: str
    degrees: List[DegreeRecord] = field(default_factory=list)
    certifications: List[CertificationRecord] = field(default_factory=list)
    degrees_found: int = 0
    certifications_found: int = 0
    highest_degree: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    status: str = "success"  # "success" | "partial" | "failed"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResultStore:
    """Writes AcademicProfileRecord objects to disk as JSON."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.json_dir = self.output_dir / "structured"
        self.json_dir.mkdir(parents=True, exist_ok=True)

    def save(self, record: AcademicProfileRecord, name_hint: str) -> str:
        stem = Path(name_hint).stem or "profile"
        json_path = self.json_dir / f"{stem}.academic_profile.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)
        return str(json_path)

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
