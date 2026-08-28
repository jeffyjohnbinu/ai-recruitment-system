"""
engine.py
---------
Top-level orchestrator for Day 15: Fairness, Normalization & Bias
Reduction. Wires together:

    raw candidate profiles
        -> ResumeNormalizer      (standardize structure)
        -> AttributeMasker       (mask non-essential personal attributes)
        -> KeywordDependencyReducer  (rebalance provided score components)
        -> ScoreNormalizer       (normalize final scores across the pool)
        -> BiasAuditor           (evaluate bias indicators, if group labels given)

This module is an additive layer: it consumes score components exactly
as produced by the Day 13 ATS Scoring Engine's WeightProfile and
candidate profiles as assembled from Day 5-11 outputs, without modifying
any upstream module or schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .bias_auditor import BiasAuditor, BiasAuditReport
from .keyword_balancer import KeywordBalanceReport, KeywordDependencyReducer
from .masker import AttributeMasker, MaskingResult
from .normalizer import CanonicalCandidateProfile, ResumeNormalizer
from .score_normalizer import NormalizedScoreRecord, ScoreNormalizer
from .storage import FairnessProcessingRecord, FairnessResultStore


@dataclass
class CandidateFairnessResult:
    candidate_id: str
    canonical_profile: CanonicalCandidateProfile
    masking: MaskingResult
    keyword_balance: KeywordBalanceReport
    adjusted_final_score: float


@dataclass
class FairnessProcessingResult:
    role_id: Optional[str]
    per_candidate: List[CandidateFairnessResult]
    normalized_scores: List[NormalizedScoreRecord]
    bias_reports: List[BiasAuditReport]

    def to_storage_record(self) -> FairnessProcessingRecord:
        masked_profiles = [
            {
                "candidate_id": c.candidate_id,
                **c.masking.to_dict(include_reversible_map=False),
                "canonical_profile": c.canonical_profile.to_dict(),
            }
            for c in self.per_candidate
        ]
        keyword_reports = {c.candidate_id: c.keyword_balance.to_dict() for c in self.per_candidate}
        return FairnessProcessingRecord(
            generated_at=FairnessResultStore.now_iso(),
            role_id=self.role_id,
            candidate_count=len(self.per_candidate),
            masked_profiles=masked_profiles,
            keyword_balance_reports=keyword_reports,
            normalized_scores=[s.to_dict() for s in self.normalized_scores],
            bias_reports=[r.to_dict() for r in self.bias_reports],
        )


class FairnessBiasEngine:
    def __init__(
        self,
        max_keyword_share: float = 0.35,
        diminishing_threshold: int = 8,
        mask_institution_names: bool = True,
        neutralize_pronouns: bool = True,
    ):
        self.normalizer = ResumeNormalizer()
        self.masker = AttributeMasker(
            mask_institution_names=mask_institution_names,
            neutralize_pronouns=neutralize_pronouns,
        )
        self.keyword_reducer = KeywordDependencyReducer(
            max_keyword_share=max_keyword_share,
            diminishing_threshold=diminishing_threshold,
        )
        self.score_normalizer = ScoreNormalizer()
        self.bias_auditor = BiasAuditor()

    def process_candidate_pool(
        self,
        raw_profiles: List[Dict[str, Any]],
        score_components: Dict[str, Dict[str, Tuple[float, float]]],
        role_id: Optional[str] = None,
        demographic_groups: Optional[Dict[str, Dict[str, str]]] = None,
        shortlist_flags: Optional[Dict[str, bool]] = None,
    ) -> FairnessProcessingResult:
        """
        raw_profiles: list of raw candidate profile dicts (must include a
            candidate identifier under one of: candidate_id/id/source_file).
        score_components: {candidate_id: {component_name: (raw_score, weight)}}
            as produced by Day 13's WeightProfile-based scoring.
        demographic_groups: optional {dimension_name: {candidate_id: group_label}}
            supplied out-of-band for audit purposes only (never used in scoring).
        shortlist_flags: optional {candidate_id: bool} from Day 14's ranking zones.
        """
        per_candidate: List[CandidateFairnessResult] = []
        raw_final_scores: Dict[str, float] = {}

        for raw_profile in raw_profiles:
            canonical = self.normalizer.normalize(raw_profile)
            cid = canonical.candidate_id

            masking = self.masker.mask(raw_profile)

            components = score_components.get(cid, {})
            balance_report = (
                self.keyword_reducer.rebalance(components)
                if components
                else KeywordBalanceReport(
                    {}, {}, None, False, 0.0, ["no score components supplied"]
                )
            )

            adjusted_final = (
                KeywordDependencyReducer.weighted_total(balance_report.adjusted_components)
                if balance_report.adjusted_components
                else 0.0
            )

            raw_final_scores[cid] = adjusted_final
            per_candidate.append(
                CandidateFairnessResult(
                    candidate_id=cid,
                    canonical_profile=canonical,
                    masking=masking,
                    keyword_balance=balance_report,
                    adjusted_final_score=adjusted_final,
                )
            )

        normalized_scores = self.score_normalizer.normalize_pool(raw_final_scores)

        bias_reports: List[BiasAuditReport] = []
        if demographic_groups:
            shortlist_flags = shortlist_flags or {}
            for dimension_name, group_map in demographic_groups.items():
                candidates_for_audit = [
                    {
                        "group": group_map.get(c.candidate_id, "unspecified"),
                        "score": c.adjusted_final_score,
                        "shortlisted": shortlist_flags.get(c.candidate_id, False),
                    }
                    for c in per_candidate
                ]
                bias_reports.append(
                    self.bias_auditor.audit_dimension(dimension_name, candidates_for_audit)
                )

        return FairnessProcessingResult(
            role_id=role_id,
            per_candidate=per_candidate,
            normalized_scores=normalized_scores,
            bias_reports=bias_reports,
        )
