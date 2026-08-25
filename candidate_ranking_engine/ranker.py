"""
Candidate Ranking & Shortlisting Engine
-----------------------------------------
Top-level orchestrator: match records -> sort -> zone -> shortlist -> storage.

Pipeline position (Day 7 architecture): consumes Day 12 Semantic Matching
Engine output and produces the recruiter-facing ranked/shortlisted view
that would feed an ATS dashboard or the next screening stage.

Usage:
    engine = CandidateRankingEngine()
    report = engine.rank_job(candidates)               # in-memory list
    report = engine.rank_directory("outputs/matches", job_id="job_123")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .loader import load_from_directory
from .storage import CandidateMatchInput, RankedCandidate, RankingReport, ResultStore

logger = logging.getLogger("candidate_ranking_engine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

ZONE_SHORTLIST = "Shortlist"
ZONE_REVIEW = "Review"
ZONE_AUTO_REJECT = "Auto-Reject"


class InvalidConfigError(ValueError):
    pass


class NoCandidatesError(ValueError):
    """Raised when there are no candidates to rank for a job."""


@dataclass
class RankingConfig:
    """
    Tunable thresholds for shortlisting. Scores are expected in [0, 1],
    consistent with the Day 12 Semantic Matching Engine's output scale.

    shortlist_threshold: score at or above this -> Shortlist zone.
    review_threshold:    score at or above this (but below shortlist) ->
                          Review zone. Below this -> Auto-Reject.
    top_n:                size of the "top candidates" convenience list.
    """

    shortlist_threshold: float = 0.75
    review_threshold: float = 0.50
    top_n: int = 10

    def __post_init__(self) -> None:
        if not (0.0 <= self.review_threshold <= self.shortlist_threshold <= 1.0):
            raise InvalidConfigError(
                "Thresholds must satisfy 0 <= review_threshold <= shortlist_threshold <= 1 "
                f"(got review={self.review_threshold}, shortlist={self.shortlist_threshold})"
            )
        if self.top_n < 1:
            raise InvalidConfigError(f"top_n must be >= 1 (got {self.top_n})")


class CandidateRankingEngine:
    def __init__(self, config: Optional[RankingConfig] = None, output_dir: str | Path = "outputs"):
        self.config = config or RankingConfig()
        self.store = ResultStore(output_dir)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def rank_directory(
        self, directory: str | Path, job_id: str, save: bool = True
    ) -> RankingReport:
        candidates = load_from_directory(directory, job_id=job_id)
        report = self.rank_job(candidates, job_id=job_id)
        if save:
            self.store.save(report)
        return report

    def rank_job(
        self, candidates: List[CandidateMatchInput], job_id: Optional[str] = None
    ) -> RankingReport:
        if not candidates:
            raise NoCandidatesError(
                f"No candidate match records found for job_id={job_id!r}; cannot produce a ranking."
            )

        resolved_job_id = job_id or candidates[0].job_id
        mismatched = [c for c in candidates if c.job_id != resolved_job_id]
        if mismatched:
            logger.warning(
                "%d candidate record(s) belong to a different job_id than %r and were excluded.",
                len(mismatched),
                resolved_job_id,
            )
            candidates = [c for c in candidates if c.job_id == resolved_job_id]

        if not candidates:
            raise NoCandidatesError(
                f"No candidate match records remain for job_id={resolved_job_id!r}."
            )

        ordered = self._sort_candidates(candidates)
        ranked = self._assign_ranks_and_zones(ordered)

        shortlist_count = sum(1 for c in ranked if c.zone == ZONE_SHORTLIST)
        review_count = sum(1 for c in ranked if c.zone == ZONE_REVIEW)
        reject_count = sum(1 for c in ranked if c.zone == ZONE_AUTO_REJECT)

        report = RankingReport(
            job_id=resolved_job_id,
            generated_at=ResultStore.now_iso(),
            total_candidates=len(ranked),
            shortlist_count=shortlist_count,
            review_count=review_count,
            auto_reject_count=reject_count,
            shortlist_threshold=self.config.shortlist_threshold,
            review_threshold=self.config.review_threshold,
            top_n=self.config.top_n,
            top_candidates=ranked[: self.config.top_n],
            ranked_candidates=ranked,
        )
        logger.info(
            "Ranked %d candidate(s) for job=%s: %d shortlist / %d review / %d auto-reject",
            len(ranked),
            resolved_job_id,
            shortlist_count,
            review_count,
            reject_count,
        )
        return report

    def save(self, report: RankingReport) -> dict:
        return self.store.save(report)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _sort_candidates(self, candidates: List[CandidateMatchInput]) -> List[CandidateMatchInput]:
        """
        Sort descending by overall_score. Ties are broken deterministically
        using the 'skills' section score if present, then candidate_id, so
        re-running the ranker on unchanged input always yields the same order.
        """
        return sorted(
            candidates,
            key=lambda c: (
                -c.overall_score,
                -c.section_scores.get("skills", 0.0),
                c.candidate_id,
            ),
        )

    def _assign_ranks_and_zones(self, ordered: List[CandidateMatchInput]) -> List[RankedCandidate]:
        timestamp = ResultStore.now_iso()
        ranked: List[RankedCandidate] = []
        for i, c in enumerate(ordered, start=1):
            zone = self.classify_zone(c.overall_score)
            ranked.append(
                RankedCandidate(
                    rank=i,
                    candidate_id=c.candidate_id,
                    job_id=c.job_id,
                    candidate_name=c.candidate_name,
                    overall_score=round(c.overall_score, 4),
                    band=c.band,
                    zone=zone,
                    section_scores=c.section_scores,
                    reason=self._explain(c, zone),
                    generated_at=timestamp,
                )
            )
        return ranked

    def classify_zone(self, score: float) -> str:
        """
        Map a similarity score to a shortlisting zone using this engine's
        configured thresholds (independent of any upstream 'band' label,
        which callers may still surface for context).
        """
        if score >= self.config.shortlist_threshold:
            return ZONE_SHORTLIST
        if score >= self.config.review_threshold:
            return ZONE_REVIEW
        return ZONE_AUTO_REJECT

    def _explain(self, candidate: CandidateMatchInput, zone: str) -> str:
        pct = f"{candidate.overall_score * 100:.1f}%"

        if zone == ZONE_SHORTLIST:
            return (
                f"Score {pct} meets shortlist threshold "
                f"({self.config.shortlist_threshold * 100:.0f}%)."
            )

        if zone == ZONE_REVIEW:
            return (
                f"Score {pct} is between review "
                f"({self.config.review_threshold * 100:.0f}%) and shortlist "
                f"({self.config.shortlist_threshold * 100:.0f}%) thresholds; "
                "recommend manual review."
            )

        return (
            f"Score {pct} is below the review threshold "
            f"({self.config.review_threshold * 100:.0f}%)."
        )
