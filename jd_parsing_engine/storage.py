"""
Structured Storage
-------------------
Defines the "Job Requirement Object" — the structured, AI-friendly
representation of a parsed job description — and persists it as JSON
(for pipeline consumption, e.g. ats_engine matching) plus a plain
cleaned .txt (for human review).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class JobRequirementRecord:
    source_file: str
    extracted_at: str

    # --- Role ---
    raw_title: Optional[str]
    normalized_role: Optional[str]
    seniority_level: Optional[str]

    # --- Skills ---
    required_skills: List[str]
    preferred_skills: List[str]

    # --- Experience ---
    min_experience_years: Optional[float]
    max_experience_years: Optional[float]
    experience_mentions: List[str]

    # --- Education ---
    education_level: Optional[str]
    education_fields: List[str]
    education_mentions: List[str]

    # --- Text / metadata ---
    raw_char_count: int
    cleaned_char_count: int
    sections_detected: List[str]
    warnings: List[str]
    cleaning_stats: Dict[str, Any]
    cleaned_text: str
    status: str = "success"  # "success" | "partial" | "failed"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_ai_profile(self) -> Dict[str, Any]:
        """
        A trimmed-down view built specifically for AI-consumption in
        prompts (screening_ai / interview_ai): drops metadata that has
        no bearing on candidate matching, keeps everything that does.
        """
        return {
            "role": self.normalized_role or self.raw_title,
            "seniority_level": self.seniority_level,
            "required_skills": self.required_skills,
            "preferred_skills": self.preferred_skills,
            "min_experience_years": self.min_experience_years,
            "max_experience_years": self.max_experience_years,
            "education_level": self.education_level,
            "education_fields": self.education_fields,
        }


class JDResultStore:
    """Writes JobRequirementRecord objects to disk as JSON + cleaned .txt."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.json_dir = self.output_dir / "structured"
        self.text_dir = self.output_dir / "cleaned_text"
        self.profile_dir = self.output_dir / "ai_profiles"
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.text_dir.mkdir(parents=True, exist_ok=True)
        self.profile_dir.mkdir(parents=True, exist_ok=True)

    def save(self, record: JobRequirementRecord) -> Dict[str, str]:
        stem = Path(record.source_file).stem
        json_path = self.json_dir / f"{stem}.json"
        text_path = self.text_dir / f"{stem}.clean.txt"
        profile_path = self.profile_dir / f"{stem}.profile.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)

        with open(text_path, "w", encoding="utf-8") as f:
            f.write(record.cleaned_text)

        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(record.to_ai_profile(), f, indent=2, ensure_ascii=False)

        return {"json": str(json_path), "text": str(text_path), "profile": str(profile_path)}

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
