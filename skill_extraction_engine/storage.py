"""
storage.py
-----------
Persists SkillExtractionResult objects as structured JSON, following the
same output convention as Day 5 (resume_extraction_engine) and Day 6
(jd_parsing_engine) so downstream pipeline stages (ATS matching, AI
screening) can consume any of them the same way.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .normalizer import SkillRecord


@dataclass
class SkillExtractionResult:
    source_file: str
    extracted_at: str
    input_char_count: int
    total_skills_found: int
    skills_by_category: Dict[str, int]
    skills: List[SkillRecord]
    status: str = "success"  # "success" | "partial" | "failed"
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


class SkillResultStore:
    """Writes SkillExtractionResult objects to disk as structured JSON."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.json_dir = self.output_dir / "structured"
        self.json_dir.mkdir(parents=True, exist_ok=True)

    def save(self, result: SkillExtractionResult, name_hint: str = "skills") -> str:
        stem = Path(name_hint).stem or "skills"
        json_path = self.json_dir / f"{stem}.skills.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        return str(json_path)

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
