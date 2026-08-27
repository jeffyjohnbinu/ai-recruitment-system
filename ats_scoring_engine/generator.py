"""
generator.py
-------------
Candidate Score Generator (Day 13 deliverable #3).

`ATSScoringEngine` computes one score; `WeightProfileRegistry` configures how
it's weighted. This module is the piece that actually *generates* candidate
scores at the volume a recruiter or a downstream pipeline stage (e.g. Day 14
ranking) needs: one call per candidate/job pair, a batch call across many,
or a full JSON-manifest-driven batch run that also persists every record and
produces a recruiter-ready leaderboard.

Following the additive-only integration convention, this module only wraps
`ATSScoringEngine` + `storage.py` -- it does not change either.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .scoring_engine import ATSScoringEngine
from .storage import ResultStore, ScoreRecord, build_score_record


@dataclass
class CandidateScoreRequest:
    """One candidate/job pair's worth of inputs to be scored."""

    candidate_id: str
    job_id: str
    role: Optional[str] = None
    skill_data: Optional[dict] = None
    experience_data: Optional[dict] = None
    education_data: Optional[dict] = None
    semantic_data: Optional[dict] = None
    score_overrides: Optional[Dict[str, float]] = None


def _resolve_field(entry: dict, base_dir: Path, data_key: str, json_key: str) -> Optional[dict]:
    """
    Resolve one component's data for a manifest entry: prefer inline data
    (e.g. `skill_data`) if present, otherwise load it from a file path
    (e.g. `skill_json`) resolved relative to the manifest file's directory.
    Returns None (component unavailable) if neither is present or the file
    doesn't exist -- the scoring engine's missing-data handling takes it
    from there.
    """
    if entry.get(data_key) is not None:
        return entry[data_key]
    path_value = entry.get(json_key)
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = base_dir / path
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(manifest_path: str | Path) -> List[CandidateScoreRequest]:
    """
    Load a batch of CandidateScoreRequest objects from a JSON manifest file.

    Manifest format -- a JSON array of entries, each with `candidate_id`,
    `job_id`, optional `role`, and per-component data supplied either
    inline (`skill_data`, `experience_data`, `education_data`,
    `semantic_data`) or as a file path (`skill_json`, `experience_json`,
    `education_json`, `semantic_json`) resolved relative to the manifest
    file's own directory. A component may be omitted entirely, in which
    case the engine's configured missing-data mode applies. See
    tests/fixtures/manifest_sample.json for a working example.
    """
    manifest_path = Path(manifest_path)
    base_dir = manifest_path.parent
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))

    requests: List[CandidateScoreRequest] = []
    for entry in entries:
        requests.append(
            CandidateScoreRequest(
                candidate_id=entry["candidate_id"],
                job_id=entry["job_id"],
                role=entry.get("role"),
                skill_data=_resolve_field(entry, base_dir, "skill_data", "skill_json"),
                experience_data=_resolve_field(
                    entry, base_dir, "experience_data", "experience_json"
                ),
                education_data=_resolve_field(entry, base_dir, "education_data", "education_json"),
                semantic_data=_resolve_field(entry, base_dir, "semantic_data", "semantic_json"),
                score_overrides=entry.get("score_overrides"),
            )
        )
    return requests


class CandidateScoreGenerator:
    """
    Generates explainable candidate scores, one at a time or in bulk.

    This is the object pipeline code (a batch job, a recruiter dashboard,
    Day 14 ranking) should instantiate and call -- it wraps an
    `ATSScoringEngine` (the formula + weight profile) and an optional
    `ResultStore` (if provided, every generated record is persisted
    automatically).
    """

    def __init__(
        self,
        engine: Optional[ATSScoringEngine] = None,
        store: Optional[ResultStore] = None,
    ):
        self.engine = engine or ATSScoringEngine()
        self.store = store

    def generate_one(self, request: CandidateScoreRequest) -> ScoreRecord:
        """Score a single candidate/job pair and return its ScoreRecord."""
        result = self.engine.compute_score(
            candidate_id=request.candidate_id,
            job_id=request.job_id,
            skill_data=request.skill_data,
            experience_data=request.experience_data,
            education_data=request.education_data,
            semantic_data=request.semantic_data,
            role=request.role,
            score_overrides=request.score_overrides,
        )
        record = build_score_record(result)
        if self.store is not None:
            self.store.save(record)
        return record

    def generate_batch(self, requests: List[CandidateScoreRequest]) -> List[ScoreRecord]:
        """Score a list of candidate/job pairs, preserving input order."""
        return [self.generate_one(r) for r in requests]

    def generate_from_manifest(self, manifest_path: str | Path) -> List[ScoreRecord]:
        """Load a JSON manifest and score every entry in it."""
        return self.generate_batch(load_manifest(manifest_path))

    @staticmethod
    def build_leaderboard(records: List[ScoreRecord]) -> List[Dict[str, Any]]:
        """
        Rank a batch of ScoreRecords best-first for recruiter review.

        Records with status="insufficient_data" (final_score=None) are
        sorted to the bottom rather than dropped, so a recruiter can still
        see who couldn't be scored, and why (via narrative_explanation).
        """

        def sort_key(record: ScoreRecord):
            return (record.final_score is None, -(record.final_score or 0.0))

        ranked = sorted(records, key=sort_key)
        return [
            {
                "rank": i,
                "candidate_id": r.candidate_id,
                "job_id": r.job_id,
                "final_score": r.final_score,
                "status": r.status,
                "role_profile": r.role_profile,
                "narrative_explanation": r.narrative_explanation,
            }
            for i, r in enumerate(ranked, start=1)
        ]

    def generate_leaderboard(self, requests: List[CandidateScoreRequest]) -> List[Dict[str, Any]]:
        """Score a batch and return it directly as a ranked leaderboard."""
        return self.build_leaderboard(self.generate_batch(requests))
