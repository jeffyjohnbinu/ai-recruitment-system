"""
Threshold Configuration & Tuning
----------------------------------
Turns a raw overall similarity score (0.0-1.0) into a categorical match
decision, and provides a small grid-search tuner that picks the threshold
maximizing F1 against a set of labeled (score, is_match) validation pairs.

Two thresholds are tracked, not one, because "match / no match" is rarely
a clean binary in recruiting -- a borderline band is genuinely useful for
human review queues:

  score >= strong_match_threshold      -> "strong_match"
  match_threshold <= score < strong    -> "possible_match"
  score <  match_threshold             -> "no_match"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

MatchBand = str  # "strong_match" | "possible_match" | "no_match"


@dataclass
class ThresholdConfig:
    match_threshold: float = 0.55
    strong_match_threshold: float = 0.75

    def __post_init__(self) -> None:
        if not (0.0 <= self.match_threshold <= 1.0):
            raise ValueError("match_threshold must be within [0, 1]")
        if not (0.0 <= self.strong_match_threshold <= 1.0):
            raise ValueError("strong_match_threshold must be within [0, 1]")
        if self.strong_match_threshold < self.match_threshold:
            raise ValueError("strong_match_threshold must be >= match_threshold")

    def classify(self, score: float) -> MatchBand:
        if score >= self.strong_match_threshold:
            return "strong_match"
        if score >= self.match_threshold:
            return "possible_match"
        return "no_match"

    def is_match(self, score: float) -> bool:
        return score >= self.match_threshold

    def to_dict(self) -> dict:
        return {
            "match_threshold": self.match_threshold,
            "strong_match_threshold": self.strong_match_threshold,
        }


@dataclass
class TuningResult:
    best_threshold: float
    best_f1: float
    precision: float
    recall: float
    grid: List[Tuple[float, float]]  # (threshold, f1) for every candidate tried


def _prf1(
    scores: Sequence[float], labels: Sequence[bool], threshold: float
) -> Tuple[float, float, float]:
    tp = fp = fn = 0
    for score, label in zip(scores, labels):
        predicted = score >= threshold
        if predicted and label:
            tp += 1
        elif predicted and not label:
            fp += 1
        elif not predicted and label:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def tune_match_threshold(
    scores: Sequence[float],
    labels: Sequence[bool],
    step: float = 0.01,
) -> TuningResult:
    """
    Grid-search the single `match_threshold` value (0.0 to 1.0, in `step`
    increments) that maximizes F1 against labeled validation pairs.

    scores: overall similarity scores produced by `similarity.compute_similarity`
    labels: ground-truth "should this be considered a match" booleans,
            same length/order as `scores`
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must be the same length")
    if not scores:
        raise ValueError("Cannot tune thresholds against an empty validation set.")

    grid: List[Tuple[float, float]] = []
    best_threshold = 0.5
    best_f1 = -1.0
    best_precision = 0.0
    best_recall = 0.0

    n_steps = int(round(1.0 / step)) + 1
    for i in range(n_steps):
        threshold = round(i * step, 4)
        precision, recall, f1 = _prf1(scores, labels, threshold)
        grid.append((threshold, f1))
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall

    return TuningResult(
        best_threshold=best_threshold,
        best_f1=best_f1,
        precision=best_precision,
        recall=best_recall,
        grid=grid,
    )
