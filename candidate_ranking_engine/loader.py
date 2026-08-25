"""
Match Record Loader
--------------------
Loads candidate/job match records produced by the Day 12 Semantic
Matching Engine (or compatible upstream output) into the normalized
CandidateMatchInput shape this module ranks on.

Deliberately tolerant of minor key-naming variance so that new upstream
fields never require changes here (cross-day integration discipline —
additive only).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .storage import CandidateMatchInput

logger = logging.getLogger("candidate_ranking_engine.loader")

# Accepted aliases for each logical field, checked in order.
_CANDIDATE_ID_KEYS = ["candidate_id", "candidateId", "resume_id", "id"]
_JOB_ID_KEYS = ["job_id", "jobId", "job_requirement_id"]
_SCORE_KEYS = [
    "overall_score",
    "overall_similarity",
    "final_score",
    "match_score",
    "similarity_score",
    "score",
]
_BAND_KEYS = ["band", "match_band", "classification", "match_classification"]
_NAME_KEYS = ["candidate_name", "name", "full_name"]
_SECTION_SCORE_KEYS = [
    "section_scores",
    "weighted_scores",
    "component_scores",
    "breakdown",
]


class MatchRecordError(ValueError):
    """Raised when a record is missing required fields (candidate_id/job_id/score)."""


def _first_present(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def parse_match_record(
    data: Dict[str, Any], source_file: Optional[str] = None
) -> CandidateMatchInput:
    candidate_id = _first_present(data, _CANDIDATE_ID_KEYS)
    job_id = _first_present(data, _JOB_ID_KEYS)
    score = _first_present(data, _SCORE_KEYS)

    if candidate_id is None or job_id is None or score is None:
        raise MatchRecordError(
            f"Record missing required field(s) (candidate_id/job_id/score): "
            f"source={source_file!r} keys={list(data.keys())}"
        )

    try:
        score = float(score)
    except (TypeError, ValueError) as exc:
        raise MatchRecordError(
            f"Non-numeric score in record from {source_file!r}: {score!r}"
        ) from exc

    band = _first_present(data, _BAND_KEYS)
    name = _first_present(data, _NAME_KEYS)
    section_scores = _first_present(data, _SECTION_SCORE_KEYS) or {}
    if not isinstance(section_scores, dict):
        section_scores = {}

    return CandidateMatchInput(
        candidate_id=str(candidate_id),
        job_id=str(job_id),
        overall_score=score,
        band=str(band) if band is not None else None,
        section_scores={k: float(v) for k, v in section_scores.items() if _is_numeric(v)},
        candidate_name=str(name) if name is not None else None,
        source_file=source_file,
        raw=data,
    )


def _is_numeric(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def load_from_file(path: str | Path) -> CandidateMatchInput:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return parse_match_record(data, source_file=str(path))


def load_from_directory(
    directory: str | Path, job_id: Optional[str] = None
) -> List[CandidateMatchInput]:
    """
    Load every *.json record in `directory`. If `job_id` is given, only
    records matching that job are returned (records for other jobs are
    skipped, not treated as errors, since a directory of match outputs
    commonly spans several jobs).
    """
    directory = Path(directory)
    records: List[CandidateMatchInput] = []
    skipped = 0

    for file in sorted(directory.glob("*.json")):
        try:
            record = load_from_file(file)
        except (MatchRecordError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unparseable match record %s: %s", file, exc)
            skipped += 1
            continue

        if job_id is not None and record.job_id != job_id:
            continue

        records.append(record)

    if skipped:
        logger.info("Skipped %d unparseable file(s) in %s", skipped, directory)

    return records
