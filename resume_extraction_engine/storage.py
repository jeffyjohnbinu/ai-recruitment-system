"""
Structured Storage
-------------------
Persists extraction results as structured JSON (for AI pipeline consumption)
and as plain cleaned .txt (for human review / debugging).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExtractionRecord:
    source_file: str
    file_type: str  # "pdf" | "docx"
    engine_used: str
    extracted_at: str
    raw_char_count: int
    cleaned_char_count: int
    sections_detected: List[str]
    tables_found: int
    images_found: int
    warnings: List[str]
    cleaning_stats: Dict[str, Any]
    cleaned_text: str
    status: str = "success"  # "success" | "partial" | "failed"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResultStore:
    """Writes ExtractionRecord objects to disk as JSON + cleaned .txt."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.json_dir = self.output_dir / "structured"
        self.text_dir = self.output_dir / "cleaned_text"
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.text_dir.mkdir(parents=True, exist_ok=True)

    def save(self, record: ExtractionRecord) -> Dict[str, str]:
        stem = Path(record.source_file).stem
        json_path = self.json_dir / f"{stem}.json"
        text_path = self.text_dir / f"{stem}.clean.txt"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)

        with open(text_path, "w", encoding="utf-8") as f:
            f.write(record.cleaned_text)

        return {"json": str(json_path), "text": str(text_path)}

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
