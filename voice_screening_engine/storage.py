"""
storage.py
----------
Day 23 deliverable — Zecpath AI Job Portal

Persists VoiceTranscriptRecord and AIScreeningRecord objects as
structured JSON, following the same output-directory / naming
conventions used since Day 5 (`outputs/structured/<name>.json`) so
downstream tooling and CI artifact uploads treat this module's output
the same way as every other day's.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from .schema import AIScreeningRecord, VoiceTranscriptRecord


class TranscriptResultStore:
    """
    Writes voice transcript and AI screening records to disk.

    Layout:
        outputs/transcripts/<candidate_id>__<job_id>__<session_id>.json
        outputs/structured/<candidate_id>__<job_id>__<session_id>.screening.json
    """

    def __init__(self, output_dir: str | Path = "outputs"):
        self.output_dir = Path(output_dir)
        self.transcript_dir = self.output_dir / "transcripts"
        self.screening_dir = self.output_dir / "structured"
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        self.screening_dir.mkdir(parents=True, exist_ok=True)

    def save_transcript(self, record: VoiceTranscriptRecord) -> str:
        filename = f"{record.candidate_id}__{record.job_id}__{record.session_id}.json"
        path = self.transcript_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)
        return str(path)

    def save_screening(self, record: AIScreeningRecord) -> str:
        filename = f"{record.candidate_id}__{record.job_id}__{record.session_id}.screening.json"
        path = self.screening_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)
        return str(path)

    def save_pair(
        self, transcript: VoiceTranscriptRecord, screening: AIScreeningRecord
    ) -> Dict[str, str]:
        return {
            "transcript": self.save_transcript(transcript),
            "screening": self.save_screening(screening),
        }

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def new_request_id() -> str:
        return str(uuid.uuid4())
