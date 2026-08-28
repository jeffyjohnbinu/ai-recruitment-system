"""
score_normalizer.py
--------------------
Normalizes final candidate scores across a pool (typically: all
candidates for one role) so scores are comparable regardless of the
raw scoring scale's quirks -- e.g. one role's scores clustering tightly
around 0.6-0.7 while another's spread 0.2-0.9. Without pool-level
normalization, a "70%" can mean top-of-pool for one role and
middle-of-pool for another, which quietly distorts cross-role fairness
(e.g. auto-reject thresholds set from Day 14's zone-based ranking).

Three normalization views are computed for each candidate:
    - z_score:            (raw - mean) / stdev            (0 if stdev == 0)
    - percentile:         fraction of pool scoring <= this candidate
    - robust_normalized:  min-max rescale using median/IQR bounds,
                           clipped to [0, 1], resistant to a few
                           extreme outlier scores skewing the whole pool

Edge cases handled explicitly:
    - empty pool -> empty result
    - single-candidate pool -> all normalized views default to the
      candidate's raw score (no meaningful spread to normalize against)
    - zero-variance pool (all identical scores) -> z_score 0.0,
      percentile 0.5 for everyone, robust_normalized equal to raw score
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Dict, List

SCHEMA_VERSION = "1.0.0"


@dataclass
class NormalizedScoreRecord:
    candidate_id: str
    raw_score: float
    z_score: float
    percentile: float
    robust_normalized: float

    def to_dict(self) -> Dict[str, float | str]:
        return {
            "candidate_id": self.candidate_id,
            "raw_score": self.raw_score,
            "z_score": self.z_score,
            "percentile": self.percentile,
            "robust_normalized": self.robust_normalized,
        }


class ScoreNormalizer:
    """Normalizes a pool of (candidate_id, raw_score) pairs."""

    def normalize_pool(self, scores: Dict[str, float]) -> List[NormalizedScoreRecord]:
        if not scores:
            return []

        ids = list(scores.keys())
        values = [scores[i] for i in ids]
        n = len(values)

        if n == 1:
            cid = ids[0]
            v = values[0]
            return [NormalizedScoreRecord(cid, v, 0.0, 0.5, v)]

        mean = statistics.fmean(values)
        stdev = statistics.pstdev(
            values
        )  # population stdev: pool is the whole population, not a sample
        sorted_values = sorted(values)
        median = statistics.median(values)
        q1, q3 = self._quartiles(sorted_values)
        iqr = q3 - q1

        records: List[NormalizedScoreRecord] = []
        for cid in ids:
            raw = scores[cid]
            z = 0.0 if stdev == 0 else (raw - mean) / stdev
            percentile = self._percentile_rank(raw, sorted_values)
            robust = self._robust_normalize(raw, median, iqr)
            records.append(
                NormalizedScoreRecord(cid, raw, round(z, 4), round(percentile, 4), round(robust, 4))
            )

        return records

    # ------------------------------------------------------------------ #
    @staticmethod
    def _percentile_rank(value: float, sorted_values: List[float]) -> float:
        n = len(sorted_values)
        if n <= 1:
            return 0.5
        count_leq = sum(1 for v in sorted_values if v <= value)
        return count_leq / n

    @staticmethod
    def _quartiles(sorted_values: List[float]) -> tuple[float, float]:
        n = len(sorted_values)
        if n < 2:
            v = sorted_values[0] if sorted_values else 0.0
            return v, v
        try:
            quantiles = statistics.quantiles(sorted_values, n=4, method="inclusive")
            return quantiles[0], quantiles[2]
        except statistics.StatisticsError:
            v = sorted_values[0]
            return v, v

    @staticmethod
    def _robust_normalize(value: float, median: float, iqr: float) -> float:
        if iqr == 0:
            # zero-variance (or near-zero) pool: nothing meaningful to rescale against
            return max(0.0, min(1.0, value))
        # Center on median, scale by IQR, then map roughly onto [0, 1] via a
        # logistic-style squash so extreme outliers compress instead of
        # blowing the scale out for everyone else.
        scaled = (value - median) / iqr
        squashed = 1 / (1 + pow(2.718281828, -scaled))
        return max(0.0, min(1.0, squashed))
