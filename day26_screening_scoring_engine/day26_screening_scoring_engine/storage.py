"""
storage.py
----------
Day 26 deliverable — Zecpath AI Job Portal

Persists ScreeningScoringFormat and SessionScore objects as structured JSON,
following the same output-directory / naming conventions used since Day 5
(`outputs/structured/<name>.json`) so downstream tooling and CI artifact
uploads treat this module's output the same way as every other day's.

Layout:
    outputs/
      structured/
        screening_scores/
          <candidate_id>__<job_id>__<session_id>.score.json
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .scoring_format import ScreenAnswerScore, SessionScore


class ScreeningScoringStore:
    """
    Writes screening score records to disk.

    Layout:
        outputs/structured/screening_scores/<candidate_id>__<job_id>__<session_id>.score.json
    """

    def __init__(self, output_dir: str | Path = "outputs") -> None:
        self.output_dir = Path(output_dir)
        self.score_dir = self.output_dir / "structured" / "screening_scores"
        self.score_dir.mkdir(parents=True, exist_ok=True)

    def save_session(self, session: SessionScore) -> str:
        """Persist a SessionScore to JSON. Returns the absolute path."""
        filename = f"{session.candidate_id}__{session.job_id}__{session.session_id}.score.json"
        path = self.score_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
        return str(path)

    def save_answer(self, question_id: str, score: ScreenAnswerScore) -> str:
        """Persist a single question's ScreenAnswerScore (for debugging / per-Q audit)."""
        filename = f"qscore_{question_id}.json"
        path = self.score_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(score.to_dict(), f, indent=2, ensure_ascii=False)
        return str(path)

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def new_request_id() -> str:
        return str(uuid.uuid4())
