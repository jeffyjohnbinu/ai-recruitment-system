"""
engine.py
--------
Day 27 deliverable — Zecpath AI Job Portal

ConfidenceSentimentEngine: main orchestrator that:

  1. Scores confidence per answer (hesitation, pace, uncertainty)
  2. Scores sentiment per answer (polarity, emotion signals)
  3. Detects contradictions (internal + cross-answer)
  4. Builds per-answer CommunicationStrengthIndicator
  5. Aggregates to a session-level BehavioralIndicatorsReport

Public API:
    engine = ConfidenceSentimentEngine()
    result = engine.score_session(candidate_id, job_id, session_id, role_id, answers)

    engine.score_answer(question_id, answer_record)   # per-answer
    engine.score_all(answers)                          # batch
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .confidence_analyzer import (
    analyze_confidence,
    detect_hesitation_patterns,
    detect_uncertainty,
    measure_pace,
)
from .formats import (
    MODEL_VERSION,
    PIPELINE_VERSION,
    SCHEMA_VERSION,
    BehavioralIndicatorsReport,
    CommunicationStrengthIndicator,
    ConfidenceAnalysis,
    new_request_id,
    now_iso,
)
from .sentiment_scorer import detect_contradictions, score_sentiment

# ---- per-answer score record ---- #


@dataclass
class AnswerScore:
    """
    Per-answer scoring output (internal).
    Aggregated into ConfidenceAnalysis for persistence and reporting.
    """

    question_id: str
    raw_answer: str
    category: Optional[str]
    hesitation_patterns: List[Any]  # List[HesitationPattern]
    pace_metrics: Any  # PaceMetrics
    uncertainty_signals: List[Any]  # List[UncertaintySignal]
    sentiment: Any  # SentimentScore
    strength_indicator: CommunicationStrengthIndicator
    overall_confidence_score: float
    warnings: List[str] = field(default_factory=list)
    duration_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "raw_answer": self.raw_answer,
            "category": self.category,
            "hesitation_patterns": [h.to_dict() for h in self.hesitation_patterns],
            "pace_metrics": self.pace_metrics.to_dict(),
            "uncertainty_signals": [u.to_dict() for u in self.uncertainty_signals],
            "sentiment": self.sentiment.to_dict(),
            "strength_indicator": self.strength_indicator.to_dict(),
            "overall_confidence_score": round(self.overall_confidence_score, 3),
            "warnings": self.warnings,
            "duration_seconds": self.duration_seconds,
        }


# ---- session-level result (engine output, lighter than formats) ---- #


@dataclass
class SessionConfidenceResult:
    """Lightweight session-level confidence result returned by the engine."""

    candidate_id: str
    job_id: str
    session_id: str
    role_id: str
    generated_at: str
    request_id: str

    per_answer: List[AnswerScore]

    # Aggregated session signals.
    session_hesitation_rate: float
    session_uncertainty_rate: float
    session_avg_confidence: float
    session_avg_strength: float
    session_sentiment_polarity: float
    session_sentiment_label: str
    strength_distribution: Dict[str, int]
    avg_response_length: float
    avg_pace_wps: Optional[float]
    avg_filler_ratio: float
    high_hesitation_answers: List[str]
    high_uncertainty_answers: List[str]
    low_confidence_answers: List[str]
    negative_sentiment_answers: List[str]
    contradictions_detected: List[Dict[str, Any]]
    narrative: str
    warnings: List[str]

    schema_version: str = SCHEMA_VERSION
    model_version: str = MODEL_VERSION
    pipeline_version: str = PIPELINE_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "pipeline_version": self.pipeline_version,
            "generated_at": self.generated_at,
            "request_id": self.request_id,
            "candidate_id": self.candidate_id,
            "job_id": self.job_id,
            "session_id": self.session_id,
            "role_id": self.role_id,
            "per_answer": [a.to_dict() for a in self.per_answer],
            "session_hesitation_rate": round(self.session_hesitation_rate, 3),
            "session_uncertainty_rate": round(self.session_uncertainty_rate, 3),
            "session_avg_confidence": round(self.session_avg_confidence, 3),
            "session_avg_strength": round(self.session_avg_strength, 3),
            "session_sentiment_polarity": round(self.session_sentiment_polarity, 3),
            "session_sentiment_label": self.session_sentiment_label,
            "strength_distribution": self.strength_distribution,
            "avg_response_length": round(self.avg_response_length, 2),
            "avg_pace_wps": (
                round(self.avg_pace_wps, 2) if self.avg_pace_wps is not None else None
            ),
            "avg_filler_ratio": round(self.avg_filler_ratio, 3),
            "high_hesitation_answers": self.high_hesitation_answers,
            "high_uncertainty_answers": self.high_uncertainty_answers,
            "low_confidence_answers": self.low_confidence_answers,
            "negative_sentiment_answers": self.negative_sentiment_answers,
            "contradictions_detected": self.contradictions_detected,
            "narrative": self.narrative,
            "warnings": self.warnings,
        }


# ---- Engine ---- #


class ConfidenceSentimentEngine:
    """
    Main orchestrator for confidence & sentiment analysis.

    Design:
      - Rule-based by default (all heuristics are pure functions in
        confidence_analyzer.py and sentiment_scorer.py).
      - Optional LLM re-scoring via `use_llm=True` (stub; falls back to rules).
      - Outputs use the Day 7 metadata envelope.
      - Additive: never mutates upstream answer records.
    """

    # Weight for combining per-answer confidence signals.
    # Hesitation and uncertainty are the primary behavioral concern.
    CONFIDENCE_WEIGHTS = {
        "hesitation_penalty": 0.35,
        "pace_score": 0.20,
        "length_score": 0.15,
        "uncertainty_penalty": 0.20,
        "sentiment_positivity": 0.10,
    }

    # Strength indicator dimension weights.
    STRENGTH_WEIGHTS = {
        "clarity": 0.25,
        "confidence": 0.25,
        "conviction": 0.20,
        "engagement": 0.15,
        "professionalism": 0.15,
    }

    # Strength thresholds.
    STRENGTH_THRESHOLDS = {
        "exceptional": 0.90,
        "strong": 0.75,
        "competent": 0.60,
        "developing": 0.40,
        # below 0.40 -> weak
    }

    # Per-answer threshold for flagging high hesitation.
    HIGH_HESITATION_THRESHOLD = 4
    HIGH_UNCERTAINTY_THRESHOLD = 3
    LOW_CONFIDENCE_THRESHOLD = 0.50
    NEGATIVE_SENTIMENT_THRESHOLD = 0.40

    def __init__(self, use_llm: bool = False) -> None:
        """
        Args:
            use_llm: If True, attempts LLM re-scoring for sentiment /
                     contradictions after the rule-based pass.
                     (Stub: falls back to rules.)
        """
        self.use_llm = use_llm

    # ------------------------------------------------------------------ #
    # Public entry point: score_session
    # ------------------------------------------------------------------ #
    def score_session(
        self,
        candidate_id: str,
        job_id: str,
        session_id: str,
        role_id: str,
        answers: List[Tuple[str, Dict[str, Any]]],
        *,
        request_id: Optional[str] = None,
    ) -> SessionConfidenceResult:
        """
        Score all answers in one interview session and aggregate.

        Args:
            candidate_id: Unique candidate identifier.
            job_id: The job this interview is for.
            session_id: Unique identifier for this call/session.
            role_id: e.g. "software_engineer".
            answers: List of (question_id, answer_record) tuples.
                     Each answer_record should have:
                       raw_answer: str
                       category: str              (optional)
                       duration_seconds: float   (optional)
        """
        rid = request_id or new_request_id()
        now = now_iso()

        # Per-answer scoring.
        per_answer: List[AnswerScore] = []
        for qid, rec in answers:
            score = self.score_answer(qid, rec)
            per_answer.append(score)

        # Cross-answer contradiction detection.
        cross_contradictions = self._detect_session_contradictions(per_answer)

        # Aggregations.
        agg = self._aggregate_session(
            candidate_id=candidate_id,
            job_id=job_id,
            session_id=session_id,
            role_id=role_id,
            per_answer=per_answer,
            cross_contradictions=cross_contradictions,
            request_id=rid,
            now=now,
        )

        return agg

    # ------------------------------------------------------------------ #
    # Per-answer scoring
    # ------------------------------------------------------------------ #
    def score_answer(
        self,
        question_id: str,
        answer_record: Dict[str, Any],
    ) -> AnswerScore:
        """
        Score one answer's confidence, pace, uncertainty, and sentiment.

        answer_record must have:
          raw_answer: str
          category: str              (optional)
          duration_seconds: float   (optional)
        """
        raw_answer = str(answer_record.get("raw_answer", "") or "")
        category = answer_record.get("category")
        duration_seconds = answer_record.get("duration_seconds")

        warnings: List[str] = []

        # 1. Hesitation patterns.
        hesitations = detect_hesitation_patterns(raw_answer)

        # 2. Pace metrics.
        pace = measure_pace(raw_answer, duration_seconds=duration_seconds)

        # 3. Uncertainty signals.
        uncertainties = detect_uncertainty(raw_answer, category=category)

        # 4. Sentiment.
        sentiment = score_sentiment(raw_answer, category=category)

        # 5. Strength indicator.
        strength = self._build_strength_indicator(
            hesitations=hesitations,
            pace=pace,
            uncertainties=uncertainties,
            sentiment=sentiment,
        )

        # 6. Overall confidence score.
        confidence_score = self._compute_overall_confidence(
            hesitations=hesitations,
            pace=pace,
            uncertainties=uncertainties,
            sentiment=sentiment,
        )

        # 7. Warnings.
        if len(hesitations) > self.HIGH_HESITATION_THRESHOLD:
            warnings.append(
                f"High hesitation count ({len(hesitations)}); candidate may be nervous or unclear."
            )
        if len(uncertainties) > self.HIGH_UNCERTAINTY_THRESHOLD:
            warnings.append(
                f"High uncertainty count ({len(uncertainties)}); answers lack conviction."
            )
        if pace.length_label in ("too_short", "short"):
            warnings.append(f"Answer length is '{pace.length_label}' ({pace.word_count} words).")
        if pace.pace_label in ("too_fast", "too_slow"):
            warnings.append(f"Speaking pace is '{pace.pace_label}'.")
        if sentiment.sentiment_label == "negative":
            warnings.append("Negative sentiment detected.")
        if confidence_score < self.LOW_CONFIDENCE_THRESHOLD:
            warnings.append(f"Overall confidence score is low ({confidence_score:.2f}).")

        return AnswerScore(
            question_id=question_id,
            raw_answer=raw_answer,
            category=category,
            hesitation_patterns=hesitations,
            pace_metrics=pace,
            uncertainty_signals=uncertainties,
            sentiment=sentiment,
            strength_indicator=strength,
            overall_confidence_score=round(confidence_score, 4),
            warnings=warnings,
            duration_seconds=duration_seconds,
        )

    # ------------------------------------------------------------------ #
    # Batch
    # ------------------------------------------------------------------ #
    def score_all(
        self,
        answers: List[Tuple[str, Dict[str, Any]]],
    ) -> List[AnswerScore]:
        """Score a list of (question_id, answer_record) tuples."""
        return [self.score_answer(qid, rec) for qid, rec in answers]

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _build_strength_indicator(
        self,
        hesitations: List[Any],
        pace: Any,
        uncertainties: List[Any],
        sentiment: Any,
    ) -> CommunicationStrengthIndicator:
        """Build a CommunicationStrengthIndicator from per-signal scores."""

        # 1. Clarity — based on hesitation count and pace.
        hesitation_count = len(hesitations)
        word_count = pace.word_count
        if word_count == 0:
            clarity = 0.0
        else:
            hesitation_density = hesitation_count / max(word_count, 1)
            # Lower density = higher clarity.
            hesitation_penalty = min(0.6, hesitation_density * 10)
            clarity = max(0.0, 1.0 - hesitation_penalty)
            # Factor in pace/length.
            clarity = 0.6 * clarity + 0.25 * pace.pace_score + 0.15 * pace.length_score

        # 2. Confidence — based on uncertainty count and hesitation severity.
        uncertainty_count = len(uncertainties)
        if word_count == 0:
            confidence = 0.0
        else:
            uncertainty_density = uncertainty_count / max(word_count, 1)
            # Higher uncertainty = lower confidence.
            uncertainty_penalty = min(0.7, uncertainty_density * 8)
            hesitation_severity = sum(h.severity for h in hesitations) / max(word_count, 1)
            hesitation_penalty = min(0.3, hesitation_severity * 3)
            confidence = max(0.0, 1.0 - uncertainty_penalty - hesitation_penalty)

        # 3. Conviction — based on sentiment polarity + low uncertainty.
        if word_count == 0:
            conviction = 0.0
        else:
            polarity = sentiment.polarity  # -1 to +1
            polarity_norm = (polarity + 1) / 2  # 0 to 1
            # Strong conviction = high positive polarity OR strong negative
            # polarity. We want to reward "decisiveness" (any clear polarity).
            decisiveness = abs(polarity)
            # Conviction = blend of polarity positivity AND decisiveness.
            conviction = 0.4 * polarity_norm + 0.4 * decisiveness + 0.2 * confidence

        # 4. Engagement — based on length and pace.
        if word_count == 0:
            engagement = 0.0
        else:
            # Long, appropriately-paced answers signal engagement.
            length_factor = min(1.0, word_count / 30.0)  # 30+ words = engaged
            pace_factor = pace.pace_score
            engagement = 0.7 * length_factor + 0.3 * pace_factor

        # 5. Professionalism — based on absence of severe hesitation, extreme
        # hedging, and positive sentiment.
        if word_count == 0:
            professionalism = 0.0
        else:
            # Penalize heavy hedges and fillers.
            heavy_hedge_count = sum(1 for u in uncertainties if u.severity > 0.7)
            hedge_penalty = min(0.5, heavy_hedge_count * 0.1)
            professionalism = (
                0.5 * (1.0 - hedge_penalty)
                + 0.3 * (sentiment.neutral_score + sentiment.positive_score) / 2.0
                + 0.2 * pace.pace_score
            )

        # Overall.
        dims = {
            "clarity": clarity,
            "confidence": confidence,
            "conviction": conviction,
            "engagement": engagement,
            "professionalism": professionalism,
        }
        overall = sum(dims[d] * w for d, w in self.STRENGTH_WEIGHTS.items())
        overall = max(0.0, min(1.0, overall))

        # Strength label.
        if overall >= self.STRENGTH_THRESHOLDS["exceptional"]:
            label = "exceptional"
        elif overall >= self.STRENGTH_THRESHOLDS["strong"]:
            label = "strong"
        elif overall >= self.STRENGTH_THRESHOLDS["competent"]:
            label = "competent"
        elif overall >= self.STRENGTH_THRESHOLDS["developing"]:
            label = "developing"
        else:
            label = "weak"

        # Primary strength and weakness.
        primary_strength = max(dims, key=dims.get)
        primary_weakness = min(dims, key=dims.get)

        return CommunicationStrengthIndicator(
            clarity=round(clarity, 3),
            confidence=round(confidence, 3),
            conviction=round(conviction, 3),
            engagement=round(engagement, 3),
            professionalism=round(professionalism, 3),
            overall_strength=round(overall, 3),
            strength_label=label,
            primary_strength=primary_strength,
            primary_weakness=primary_weakness,
            note="",
        )

    def _compute_overall_confidence(
        self,
        hesitations: List[Any],
        pace: Any,
        uncertainties: List[Any],
        sentiment: Any,
    ) -> float:
        """Combine signals into a single 0-1 confidence score."""
        word_count = pace.word_count
        if word_count == 0:
            return 0.0

        # Hesitation penalty (0-1; higher = worse).
        # Combines raw count (capped) and severity-per-word.
        hesitation_count = len(hesitations)
        hesitation_density = hesitation_count / max(word_count, 1)
        hesitation_severity_avg = (
            sum(h.severity for h in hesitations) / max(hesitation_count, 1) if hesitations else 0.0
        )
        # Count-based penalty: 5+ hesitations in a short answer is severe.
        count_penalty = min(0.5, hesitation_count * 0.05)
        # Density penalty.
        density_penalty = min(0.5, hesitation_density * 5)
        # Severity penalty.
        severity_penalty = min(0.4, hesitation_severity_avg * 0.5)
        hesitation_penalty = min(1.0, count_penalty + density_penalty + severity_penalty)

        # Uncertainty penalty.
        uncertainty_count = len(uncertainties)
        uncertainty_density = uncertainty_count / max(word_count, 1)
        uncertainty_severity_avg = (
            sum(u.severity for u in uncertainties) / max(uncertainty_count, 1)
            if uncertainties
            else 0.0
        )
        unc_count_penalty = min(0.5, uncertainty_count * 0.05)
        unc_density_penalty = min(0.5, uncertainty_density * 4)
        unc_severity_penalty = min(0.4, uncertainty_severity_avg * 0.5)
        uncertainty_penalty = min(
            1.0, unc_count_penalty + unc_density_penalty + unc_severity_penalty
        )

        # Sentiment positivity (0-1; higher = better).
        sentiment_positivity = max(0.0, sentiment.polarity + 1) / 2

        # Combine.
        weights = self.CONFIDENCE_WEIGHTS
        score = (
            (1.0 - hesitation_penalty) * weights["hesitation_penalty"]
            + pace.pace_score * weights["pace_score"]
            + pace.length_score * weights["length_score"]
            + (1.0 - uncertainty_penalty) * weights["uncertainty_penalty"]
            + sentiment_positivity * weights["sentiment_positivity"]
        )

        return max(0.0, min(1.0, score))

    def _detect_session_contradictions(
        self,
        per_answer: List[AnswerScore],
    ) -> List[Dict[str, Any]]:
        """Detect self-contradictions within and across all answers."""
        all_contradictions: List[Dict[str, Any]] = []

        # Internal contradictions.
        for ans in per_answer:
            internal = detect_contradictions(ans.raw_answer)
            for c in internal:
                c["question_id"] = ans.question_id
                c["scope"] = "internal"
                all_contradictions.append(c)

        # Cross-answer contradictions (lightweight keyword check).
        all_answers = [(ans.question_id, ans.raw_answer) for ans in per_answer]
        for i, ans in enumerate(per_answer):
            others = [a for j, a in enumerate(all_answers) if j != i]
            cross = detect_contradictions(ans.raw_answer, other_answers=others)
            for c in cross:
                c["question_id"] = ans.question_id
                c["scope"] = "cross_answer"
                all_contradictions.append(c)

        return all_contradictions

    def _aggregate_session(
        self,
        candidate_id: str,
        job_id: str,
        session_id: str,
        role_id: str,
        per_answer: List[AnswerScore],
        cross_contradictions: List[Dict[str, Any]],
        request_id: str,
        now: str,
    ) -> SessionConfidenceResult:
        """Aggregate per-answer scores into session-level summary."""

        n = len(per_answer)
        if n == 0:
            return self._empty_session(candidate_id, job_id, session_id, role_id, request_id, now)

        # Hesitation rate (patterns per answer).
        hes_counts = [len(a.hesitation_patterns) for a in per_answer]
        avg_hesitation = sum(hes_counts) / n

        # Uncertainty rate.
        unc_counts = [len(a.uncertainty_signals) for a in per_answer]
        avg_uncertainty = sum(unc_counts) / n

        # Avg confidence.
        confidence_scores = [a.overall_confidence_score for a in per_answer]
        avg_confidence = sum(confidence_scores) / n

        # Avg strength.
        strengths = [a.strength_indicator.overall_strength for a in per_answer]
        avg_strength = sum(strengths) / n

        # Sentiment polarity.
        polarities = [a.sentiment.polarity for a in per_answer]
        avg_polarity = sum(polarities) / n
        # Session label.
        if avg_polarity > 0.3:
            session_label = "positive"
        elif avg_polarity < -0.3:
            session_label = "negative"
        elif avg_polarity < 0.1 and avg_polarity > -0.1:
            session_label = "neutral"
        else:
            session_label = "mixed"

        # Strength distribution.
        strength_dist: Dict[str, int] = {
            "exceptional": 0,
            "strong": 0,
            "competent": 0,
            "developing": 0,
            "weak": 0,
        }
        for a in per_answer:
            label = a.strength_indicator.strength_label
            strength_dist[label] = strength_dist.get(label, 0) + 1

        # Avg response length.
        word_counts = [a.pace_metrics.word_count for a in per_answer]
        avg_length = sum(word_counts) / n

        # Avg pace wps (where available).
        wps_values = [
            a.pace_metrics.words_per_second
            for a in per_answer
            if a.pace_metrics.words_per_second is not None
        ]
        avg_wps = sum(wps_values) / len(wps_values) if wps_values else None

        # Filler ratio.
        filler_ratios = []
        for a in per_answer:
            wc = a.pace_metrics.word_count
            hes_count = len(a.hesitation_patterns)
            if wc > 0:
                filler_ratios.append(hes_count / wc)
        avg_filler_ratio = sum(filler_ratios) / max(len(filler_ratios), 1)

        # Flagged answers.
        high_hes = [
            a.question_id
            for a in per_answer
            if len(a.hesitation_patterns) > self.HIGH_HESITATION_THRESHOLD
        ]
        high_unc = [
            a.question_id
            for a in per_answer
            if len(a.uncertainty_signals) > self.HIGH_UNCERTAINTY_THRESHOLD
        ]
        low_conf = [
            a.question_id
            for a in per_answer
            if a.overall_confidence_score < self.LOW_CONFIDENCE_THRESHOLD
        ]
        neg_sent = [a.question_id for a in per_answer if a.sentiment.sentiment_label == "negative"]

        # Warnings.
        warnings: List[str] = []
        if avg_confidence < 0.5:
            warnings.append(f"Session average confidence is low ({avg_confidence:.2f}).")
        elif avg_confidence < 0.6:
            warnings.append(f"Session confidence is below normal ({avg_confidence:.2f}).")
        if avg_hesitation > 3:
            warnings.append(
                f"High session-level hesitation rate ({avg_hesitation:.1f} per answer)."
            )
        if cross_contradictions:
            warnings.append(f"{len(cross_contradictions)} contradiction(s) detected.")
        if neg_sent:
            warnings.append(f"{len(neg_sent)} answer(s) had negative sentiment.")

        # Narrative.
        narrative = self._build_session_narrative(
            n=n,
            avg_hesitation=avg_hesitation,
            avg_uncertainty=avg_uncertainty,
            avg_confidence=avg_confidence,
            avg_strength=avg_strength,
            avg_polarity=avg_polarity,
            session_label=session_label,
            strength_dist=strength_dist,
            n_contradictions=len(cross_contradictions),
        )

        return SessionConfidenceResult(
            candidate_id=candidate_id,
            job_id=job_id,
            session_id=session_id,
            role_id=role_id,
            generated_at=now,
            request_id=request_id,
            per_answer=per_answer,
            session_hesitation_rate=avg_hesitation,
            session_uncertainty_rate=avg_uncertainty,
            session_avg_confidence=avg_confidence,
            session_avg_strength=avg_strength,
            session_sentiment_polarity=avg_polarity,
            session_sentiment_label=session_label,
            strength_distribution=strength_dist,
            avg_response_length=avg_length,
            avg_pace_wps=avg_wps,
            avg_filler_ratio=avg_filler_ratio,
            high_hesitation_answers=high_hes,
            high_uncertainty_answers=high_unc,
            low_confidence_answers=low_conf,
            negative_sentiment_answers=neg_sent,
            contradictions_detected=cross_contradictions,
            narrative=narrative,
            warnings=warnings,
        )

    def _build_session_narrative(
        self,
        n: int,
        avg_hesitation: float,
        avg_uncertainty: float,
        avg_confidence: float,
        avg_strength: float,
        avg_polarity: float,
        session_label: str,
        strength_dist: Dict[str, int],
        n_contradictions: int,
    ) -> str:
        """Generate a plain-English session-level narrative."""
        parts: List[str] = []

        parts.append(
            f"Candidate answered {n} questions with session average "
            f"confidence {avg_confidence:.0%} and communication strength "
            f"{avg_strength:.0%}."
        )

        parts.append(f"Overall sentiment: {session_label} (polarity {avg_polarity:+.2f}).")

        if avg_hesitation > 3:
            parts.append(
                f"Above-average hesitation ({avg_hesitation:.1f} markers per answer) "
                "suggests possible nervousness or lack of preparation."
            )
        elif avg_hesitation > 1:
            parts.append(f"Moderate hesitation ({avg_hesitation:.1f} markers per answer).")
        else:
            parts.append("Minimal hesitation; candidate was fluent.")

        if avg_uncertainty > 3:
            parts.append(
                f"High uncertainty markers ({avg_uncertainty:.1f} per answer) "
                "indicate lack of conviction or familiarity."
            )

        if strength_dist.get("exceptional", 0) > 0:
            parts.append(
                f"{strength_dist['exceptional']} answer(s) reached exceptional communication strength."
            )
        if strength_dist.get("strong", 0) > 0:
            parts.append(f"{strength_dist['strong']} answer(s) rated strong.")
        if strength_dist.get("weak", 0) > 0:
            parts.append(f"{strength_dist['weak']} answer(s) rated weak — review carefully.")

        if n_contradictions > 0:
            parts.append(f"{n_contradictions} contradiction(s) detected; verify consistency.")

        return " ".join(parts)

    def _empty_session(
        self,
        candidate_id: str,
        job_id: str,
        session_id: str,
        role_id: str,
        request_id: str,
        now: str,
    ) -> SessionConfidenceResult:
        return SessionConfidenceResult(
            candidate_id=candidate_id,
            job_id=job_id,
            session_id=session_id,
            role_id=role_id,
            generated_at=now,
            request_id=request_id,
            per_answer=[],
            session_hesitation_rate=0.0,
            session_uncertainty_rate=0.0,
            session_avg_confidence=0.0,
            session_avg_strength=0.0,
            session_sentiment_polarity=0.0,
            session_sentiment_label="neutral",
            strength_distribution={
                "exceptional": 0,
                "strong": 0,
                "competent": 0,
                "developing": 0,
                "weak": 0,
            },
            avg_response_length=0.0,
            avg_pace_wps=None,
            avg_filler_ratio=0.0,
            high_hesitation_answers=[],
            high_uncertainty_answers=[],
            low_confidence_answers=[],
            negative_sentiment_answers=[],
            contradictions_detected=[],
            narrative="No answers were scored; insufficient data.",
            warnings=["No answers available for confidence/sentiment analysis."],
        )

    # ------------------------------------------------------------------ #
    # Build BehavioralIndicatorsReport
    # ------------------------------------------------------------------ #
    def build_behavioral_report(
        self,
        session_result: SessionConfidenceResult,
    ) -> BehavioralIndicatorsReport:
        """
        Convert a SessionConfidenceResult to a full
        BehavioralIndicatorsReport (the Day 27 deliverable).

        Useful when persisting to a database or serving via API.
        """
        per_answer_analysis: List[ConfidenceAnalysis] = []
        for ans in session_result.per_answer:
            ca = ConfidenceAnalysis(
                question_id=ans.question_id,
                raw_answer=ans.raw_answer,
                hesitation_patterns=ans.hesitation_patterns,
                pace_metrics=ans.pace_metrics,
                uncertainty_signals=ans.uncertainty_signals,
                sentiment=ans.sentiment,
                strength_indicator=ans.strength_indicator,
                hesitation_count=len(ans.hesitation_patterns),
                uncertainty_count=len(ans.uncertainty_signals),
                overall_confidence_score=ans.overall_confidence_score,
                generated_at=session_result.generated_at,
                request_id=session_result.request_id,
                warnings=ans.warnings,
            )
            per_answer_analysis.append(ca)

        return BehavioralIndicatorsReport(
            candidate_id=session_result.candidate_id,
            job_id=session_result.job_id,
            session_id=session_result.session_id,
            role_id=session_result.role_id,
            total_answers=len(per_answer_analysis),
            per_answer=per_answer_analysis,
            session_hesitation_rate=session_result.session_hesitation_rate,
            session_uncertainty_rate=session_result.session_uncertainty_rate,
            session_avg_confidence=session_result.session_avg_confidence,
            session_avg_strength=session_result.session_avg_strength,
            session_sentiment_polarity=session_result.session_sentiment_polarity,
            session_sentiment_label=session_result.session_sentiment_label,
            strength_distribution=session_result.strength_distribution,
            high_hesitation_answers=session_result.high_hesitation_answers,
            high_uncertainty_answers=session_result.high_uncertainty_answers,
            low_confidence_answers=session_result.low_confidence_answers,
            negative_sentiment_answers=session_result.negative_sentiment_answers,
            contradictions_detected=session_result.contradictions_detected,
            avg_response_length=session_result.avg_response_length,
            avg_pace_wps=session_result.avg_pace_wps,
            avg_filler_ratio=session_result.avg_filler_ratio,
            generated_at=session_result.generated_at,
            request_id=session_result.request_id,
            narrative=session_result.narrative,
            warnings=session_result.warnings,
        )


# ---- helper for converting engine output to formats.ConfidenceAnalysis ---- #


def answer_score_to_confidence_analysis(
    ans: AnswerScore,
    generated_at: str,
    request_id: str,
) -> ConfidenceAnalysis:
    """Convert an internal AnswerScore into a formats.ConfidenceAnalysis record."""
    return ConfidenceAnalysis(
        question_id=ans.question_id,
        raw_answer=ans.raw_answer,
        hesitation_patterns=ans.hesitation_patterns,
        pace_metrics=ans.pace_metrics,
        uncertainty_signals=ans.uncertainty_signals,
        sentiment=ans.sentiment,
        strength_indicator=ans.strength_indicator,
        hesitation_count=len(ans.hesitation_patterns),
        uncertainty_count=len(ans.uncertainty_signals),
        overall_confidence_score=ans.overall_confidence_score,
        generated_at=generated_at,
        request_id=request_id,
        warnings=ans.warnings,
    )
