"""
Structured Storage
-------------------
Persists section-classification results as structured JSON (for
downstream AI pipeline / matching-engine consumption — same convention
as Day 5's `ExtractionRecord` and Day 6's Job Requirement Objects) and
as a plain, human-readable labeled-text file for manual review.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SectionBlockRecord:
    label: str
    method: str  # "rule_heading" | "nlp" | "header_heuristic"
    confidence: float
    start_line: int
    end_line: int
    char_count: int
    text: str
    heading_text: Optional[str] = None
    nlp_label: Optional[str] = None
    nlp_confidence: Optional[float] = None
    flags: List[str] = field(default_factory=list)


@dataclass
class SectionClassificationRecord:
    source_file: str
    classified_at: str
    total_char_count: int
    sections_detected: List[str]
    heading_based_blocks: int
    nlp_based_blocks: int
    uncategorized_char_count: int
    blocks: List[SectionBlockRecord]
    warnings: List[str] = field(default_factory=list)
    status: str = "success"  # "success" | "partial" | "failed"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            **{k: v for k, v in asdict(self).items() if k != "blocks"},
            "blocks": [asdict(b) for b in self.blocks],
        }

    def as_matching_engine_payload(self) -> Dict[str, List[str]]:
        """
        Convenience view: canonical label -> concatenated text, in the
        same "structured object" spirit as Day 5's resume output and
        Day 6's Job Requirement Object, so a downstream matcher can
        pull `payload["Skills"]`, `payload["Experience"]`, etc.
        directly without knowing about blocks/methods/confidence.
        """
        out: Dict[str, List[str]] = {}
        for block in self.blocks:
            out.setdefault(block.label, []).append(block.text)
        return out


class ResultStore:
    """Writes SectionClassificationRecord objects to disk as JSON + labeled .txt."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.json_dir = self.output_dir / "structured"
        self.text_dir = self.output_dir / "labeled_text"
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.text_dir.mkdir(parents=True, exist_ok=True)

    def save(self, record: SectionClassificationRecord) -> Dict[str, str]:
        stem = Path(record.source_file).stem
        json_path = self.json_dir / f"{stem}.sections.json"
        text_path = self.text_dir / f"{stem}.labeled.txt"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)

        with open(text_path, "w", encoding="utf-8") as f:
            for block in record.blocks:
                f.write(
                    f"===== [{block.label}] (method={block.method}, "
                    f"confidence={block.confidence}) =====\n"
                )
                f.write(block.text.strip() + "\n\n")

        return {"json": str(json_path), "text": str(text_path)}

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
