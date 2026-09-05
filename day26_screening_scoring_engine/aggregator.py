"""
aggregator.py
-------------
Day 26 deliverable — Zecpath AI Job Portal

Aggregates per-question scores into a session-level result.

Responsibilities:
  - Compute each question's overall score from its four dimension scores.
  - Compute each question's weighted contribution to the session total.
  - Normalize the final score to [0, 1].
  - Aggregate per-dimension scores across all questions.
  - Derive a recommendation label (proceed / hold / reject / insufficient_data).
  - Build the SessionScore dataclass.
  - Generate a plain-English narrative explanation.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger
from utils.validators import clamp_score

logger = get_logger("day26_screening_scoring_engine.aggregator")


# Default weights for the four scoring dimensions.
# These can be overridden via ScoringConfig at construction time.
DEFAULT_DIMENSION_WEIGHTS = {
    "clarity": 0.15,
    "relevance": 0.40,
    "completeness": 0.30,
    "consistency": 0.15,
}

# Minimum answer score required to count that question toward the session total.
_MIN_WEIGHTED_CONTRIBUTION = 0.01

# Recommendation thresholds.
_RECOMMENDATION_THRESHOLDS = {
    "proceed": 0.75,
    "hold": 0.45,
    # Below "hold" threshold -> reject
}


@dataclass
class ScoringConfig:
    """
    Tunable parameters for the scoring engine.
    All weights must sum to 1.0.
    """

    dimension_weights: Dict[str, float]
    recommendation_thresholds: Dict[str, float]
    hard_filter_weights: Tuple[int, ...]  # weights that act as hard filters
    min_answered_questions_for_proceed: int  # must answer at least this many to recommend "proceed"

    def __post_init__(self) -> None:
        total = sum(self.dimension_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Dimension weights must sum to 1.0; got {total:.4f}")
        hard_weights = set(self.hard_filter_weights)
        if hard_weights - {1, 2, 3, 4, 5}:
            raise ValueError(
                f"hard_filter_weights must contain only values 1-5; got {hard_weights}"
            )

    @classmethod
    def with_overrides(
        cls,
        dimension_weights: Optional[Dict[str, float]] = None,
        recommendation_thresholds: Optional[Dict[str, float]] = None,
        hard_filter_weights: Optional[Tuple[int, ...]] = None,
        min_answered: Optional[int] = None,
    ) -> "ScoringConfig":
        """Return a config with specified overrides, rest from defaults."""
        dim = {**DEFAULT_DIMENSION_WEIGHTS, **(dimension_weights or {})}
        thr = {**_RECOMMENDATION_THRESHOLD_DEFAULTS, **(recommendation_thresholds or {})}
        hard = hard_filter_weights or (5,)  # weight-5 questions are always hard filters
        min_ans = min_answered if min_answered is not None else 3
        return cls(
            dimension_weights=dim,
            recommendation_thresholds=thr,
            hard_filter_weights=hard,
            min_answered_questions_for_proceed=min_ans,
        )


_RECOMMENDATION_THRESHOLD_DEFAULTS = {
    "proceed": 0.75,
    "hold": 0.45,
}


def compute_question_score(
    dimension_scores: Dict[str, float],
    weights: Dict[str, float],
) -> float:
    """
    Blend four dimension scores into a single [0, 1] question score.

    dimension_scores: {name: score} for clarity / relevance / completeness / consistency
    weights: {name: weight} — must sum to 1.0
    """
    score = sum(dimension_scores.get(name, 0.0) * weight for name, weight in weights.items())
    return clamp_score(score)


def compute_weighted_contribution(
    question_score: float,
    scoring_weight: int,
    total_weight_sum: float,
) -> float:
    """
    A question's contribution to the session total, proportional to its weight.

    contribution = (question_score * scoring_weight) / total_weight_sum
    """
    if total_weight_sum <= 0:
        return 0.0
    return clamp_score((question_score * scoring_weight) / total_weight_sum)


def compute_session_score(
    weighted_contributions: List[float],
    total_weight_sum: float,
    answered_weight_sum: float,
) -> float:
    """
    Sum of all weighted contributions, normalized by the max possible total
    so that a perfect score always yields 1.0.
    """
    if total_weight_sum <= 0:
        return 0.0
    # Each question contributes (scoring_weight / total_weight_sum) if perfect.
    # So the denominator for normalization is 1.0.
    return clamp_score(sum(weighted_contributions))


def aggregate_session(
    breakdown_records: List[Tuple[str, Dict[str, Any]]],
    config: Optional[ScoringConfig] = None,
    *,
    candidate_id: str,
    job_id: str,
    session_id: str,
    role_id: str,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a full SessionScore dict from per-question ScreenAnswerScore dicts.

    breakdown_records: list of (question_id, screen_answer_score_dict)
    config: ScoringConfig; defaults to DEFAULT_DIMENSION_WEIGHTS
    """
    config = config or ScoringConfig.with_overrides()
    cfg = config
    now = datetime.now(timezone.utc).isoformat()
    rid = request_id or str(uuid.uuid4())

    if not breakdown_records:
        return _empty_session(candidate_id, job_id, session_id, role_id, rid, now)

    # --- Compute per-dimension session averages ---
    dim_names = list(cfg.dimension_weights.keys())
    dim_sums: Dict[str, float] = {d: 0.0 for d in dim_names}
    dim_counts: Dict[str, int] = {d: 0 for d in dim_names}
    hard_filters_failed: List[str] = []
    mandatory_unanswered: int = 0

    weighted_contributions: List[float] = []
    max_weight_sum = 0.0
    answered_weight_sum = 0.0

    breakdowns_list: List[Dict[str, Any]] = []

    for qid, rec in breakdown_records:
        weight = int(rec.get("scoring_weight", 1))
        is_mandatory = bool(rec.get("is_mandatory", False))
        was_answered = bool(rec.get("was_answered", False))
        q_score = float(rec.get("overall_score", 0.0))
        max_weight_sum += weight
        # Normalize dimension_scores: accept either a dict {name: score}
        # or a list of {name, score, ...} dicts (as serialized by the engine).
        dim_scores_raw = rec.get("dimension_scores", {})
        if isinstance(dim_scores_raw, list):
            dim_scores: Dict[str, float] = {d["name"]: float(d["score"]) for d in dim_scores_raw}
        elif isinstance(dim_scores_raw, dict):
            dim_scores = {k: float(v) for k, v in dim_scores_raw.items()}
        else:
            dim_scores = {}

        for dname in dim_names:
            ds = dim_scores.get(dname, 0.0)
            dim_sums[dname] += ds
            dim_counts[dname] += 1

        wc = compute_weighted_contribution(q_score, weight, max_weight_sum)
        weighted_contributions.append(wc)

        if was_answered:
            answered_weight_sum += weight
            if weight in cfg.hard_filter_weights and q_score < 0.5:
                hard_filters_failed.append(qid)
        else:
            if is_mandatory:
                mandatory_unanswered += 1

        breakdowns_list.append(
            {
                "question_id": qid,
                "category": rec.get("category", "unknown"),
                "scoring_weight": weight,
                "is_mandatory": is_mandatory,
                "was_answered": was_answered,
                "overall_score": round(q_score, 4),
                "weighted_contribution": round(wc, 4),
                "dimension_scores": {k: round(v, 4) for k, v in dim_scores.items()},
                "explanation": rec.get("explanation", ""),
            }
        )

    total_questions = len(breakdown_records)
    answered_questions = sum(1 for _, r in breakdown_records if r.get("was_answered"))

    # Session overall score = sum of weighted contributions (already 0..1 scaled)
    normalized_score = clamp_score(sum(weighted_contributions))

    # Per-dimension session average.
    overall_dim_scores: Dict[str, float] = {}
    for dname in dim_names:
        if dim_counts[dname] > 0:
            overall_dim_scores[dname] = dim_sums[dname] / dim_counts[dname]
        else:
            overall_dim_scores[dname] = 0.0

    # Recommendation.
    recommendation = _derive_recommendation(
        normalized_score,
        mandatory_unanswered,
        hard_filters_failed,
        answered_questions,
        total_questions,
        cfg,
    )

    # Narrative.
    narrative = _build_narrative(
        normalized_score=normalized_score,
        total_questions=total_questions,
        answered_questions=answered_questions,
        mandatory_unanswered=mandatory_unanswered,
        hard_filters_failed=hard_filters_failed,
        overall_dim_scores=overall_dim_scores,
        breakdowns=breakdowns_list,
        config=cfg,
    )

    return {
        "candidate_id": candidate_id,
        "job_id": job_id,
        "session_id": session_id,
        "generated_at": now,
        "request_id": rid,
        "role_id": role_id,
        "total_questions": total_questions,
        "answered_questions": answered_questions,
        "mandatory_unanswered": mandatory_unanswered,
        "sum_of_weights_answered": round(answered_weight_sum, 2),
        "max_possible_weight": round(max_weight_sum, 2),
        "normalized_score": round(normalized_score, 4),
        "overall_dimension_scores": {k: round(v, 4) for k, v in overall_dim_scores.items()},
        "hard_filters_failed": hard_filters_failed,
        "recommendation": recommendation,
        "consistency": {
            "is_consistent": len(hard_filters_failed) == 0,
            "score": 1.0 - 0.2 * len(hard_filters_failed),
            "notes": [],
        },
        "breakdown": breakdowns_list,
        "narrative": narrative,
        "warnings": _build_warnings(
            mandatory_unanswered, hard_filters_failed, answered_questions, total_questions
        ),
    }


def _derive_recommendation(
    normalized_score: float,
    mandatory_unanswered: int,
    hard_filters_failed: List[str],
    answered_questions: int,
    total_questions: int,
    config: ScoringConfig,
) -> str:
    """Decide the recommendation label.

    Order of precedence (highest first):
      1. No answers at all                          -> insufficient_data
      2. Hard filters failed                        -> reject
      3. Mandatory questions unanswered            -> reject
      4. Below the minimum-answered threshold       -> insufficient_data
      5. Score >= proceed_threshold                 -> proceed
      6. Score >= hold_threshold                    -> hold
      7. Otherwise                                  -> reject
    """
    thresholds = config.recommendation_thresholds

    if answered_questions == 0:
        return "insufficient_data"

    if hard_filters_failed:
        return "reject"

    if mandatory_unanswered > 0:
        return "reject"

    if answered_questions < config.min_answered_questions_for_proceed:
        return "insufficient_data"

    if normalized_score >= thresholds.get("proceed", 0.75):
        return "proceed"
    if normalized_score >= thresholds.get("hold", 0.45):
        return "hold"
    return "reject"


def _build_narrative(
    normalized_score: float,
    total_questions: int,
    answered_questions: int,
    mandatory_unanswered: int,
    hard_filters_failed: List[str],
    overall_dim_scores: Dict[str, float],
    breakdowns: List[Dict[str, Any]],
    config: ScoringConfig,
) -> str:
    """Generate a plain-English explanation of the session score."""
    parts: List[str] = []

    parts.append(
        f"Candidate answered {answered_questions}/{total_questions} questions"
        f" (score: {normalized_score:.0%})."
    )

    if mandatory_unanswered > 0:
        parts.append(
            f"{mandatory_unanswered} mandatory question(s) left unanswered — "
            "this is a concern for mandatory-answer roles."
        )

    if hard_filters_failed:
        parts.append(
            f"{len(hard_filters_failed)} critical (weight-5) question(s) "
            f"scored below 50%: {', '.join(hard_filters_failed)}."
        )

    # Weakest dimension.
    if overall_dim_scores:
        weakest = min(overall_dim_scores, key=overall_dim_scores.get)
        weakest_score = overall_dim_scores.get(weakest, 0)
        if weakest_score < 0.6:
            parts.append(f"Lowest dimension: '{weakest}' ({weakest_score:.0%}).")

    # Strongest dimension.
    if overall_dim_scores:
        strongest = max(overall_dim_scores, key=overall_dim_scores.get)
        strongest_score = overall_dim_scores.get(strongest, 0)
        if strongest_score >= 0.75:
            parts.append(f"Strongest dimension: '{strongest}' ({strongest_score:.0%}).")

    # Per-category summary.
    cat_scores: Dict[str, List[float]] = {}
    for b in breakdowns:
        cat = b.get("category", "unknown")
        cat_scores.setdefault(cat, []).append(b.get("overall_score", 0.0))

    cat_summaries = []
    for cat, scores in sorted(cat_scores.items()):
        avg = sum(scores) / len(scores)
        cat_summaries.append(f"{cat} ({avg:.0%})")
    if cat_summaries:
        parts.append("Category averages: " + "; ".join(cat_summaries) + ".")

    return " ".join(parts)


def _build_warnings(
    mandatory_unanswered: int,
    hard_filters_failed: List[str],
    answered_questions: int,
    total_questions: int,
) -> List[str]:
    warnings: List[str] = []
    if mandatory_unanswered > 0:
        warnings.append(f"{mandatory_unanswered} mandatory question(s) unanswered.")
    if hard_filters_failed:
        warnings.append(f"Hard-filter failed: {', '.join(hard_filters_failed)}.")
    if answered_questions == 0:
        warnings.append("No questions were answered — insufficient data.")
    elif answered_questions < total_questions * 0.5:
        warnings.append(f"Only {answered_questions}/{total_questions} questions answered.")
    return warnings


def _empty_session(
    candidate_id: str,
    job_id: str,
    session_id: str,
    role_id: str,
    request_id: str,
    generated_at: str,
) -> Dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "job_id": job_id,
        "session_id": session_id,
        "generated_at": generated_at,
        "request_id": request_id,
        "role_id": role_id,
        "total_questions": 0,
        "answered_questions": 0,
        "mandatory_unanswered": 0,
        "sum_of_weights_answered": 0.0,
        "max_possible_weight": 0.0,
        "normalized_score": 0.0,
        "overall_dimension_scores": {},
        "hard_filters_failed": [],
        "recommendation": "insufficient_data",
        "consistency": {"is_consistent": True, "score": 1.0, "notes": []},
        "breakdown": [],
        "narrative": "No questions were scored; insufficient data.",
        "warnings": ["No answers available for scoring."],
    }


def classify_recommendation(
    normalized_score: float,
    *,
    proceed_threshold: float = 0.75,
    hold_threshold: float = 0.45,
) -> str:
    """
    Standalone recommendation classifier.
    Returns: "proceed" | "hold" | "reject"
    """
    if normalized_score >= proceed_threshold:
        return "proceed"
    if normalized_score >= hold_threshold:
        return "hold"
    return "reject"
