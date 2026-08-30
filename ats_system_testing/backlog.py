"""
backlog.py
-----------
Turns mismatch patterns from a MetricsReport into a prioritized
improvement backlog. This is a deliberately simple, rule-based
prioritizer (frequency + severity), consistent with the project's
"explainability over black-box scores" principle -- every backlog item
traces back to specific case IDs, not an opaque ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .metrics import MetricsReport


@dataclass
class BacklogItem:
    priority: str  # "P0" | "P1" | "P2"
    title: str
    description: str
    affected_cases: List[str]
    recommended_owner: str


def build_backlog(metrics: MetricsReport) -> List[BacklogItem]:
    items: List[BacklogItem] = []
    mismatches = metrics.mismatches

    false_positives = [m for m in mismatches if m.mismatch_type == "false_positive"]
    false_negatives = [m for m in mismatches if m.mismatch_type == "false_negative"]
    rec_only = [m for m in mismatches if m.mismatch_type == "recommendation_only"]

    if false_positives:
        items.append(
            BacklogItem(
                priority="P0",
                title="Reduce false-positive shortlisting on domain-mismatched candidates",
                description=(
                    f"{len(false_positives)} case(s) where the pipeline shortlisted a "
                    "candidate a human reviewer rejected, typically because keyword "
                    "overlap alone doesn't capture that the candidate's domain "
                    "(e.g. sales vs. HR, design vs. backend engineering) differs "
                    "from the role. Recommend strengthening the Day 12 semantic "
                    "matcher's domain/role-family signal (or adding one) ahead of "
                    "raw keyword overlap in the Day 13 scoring weights."
                ),
                affected_cases=[m.case_id for m in false_positives],
                recommended_owner="Semantic Matching Engine (Day 12) / ATS Scoring Engine (Day 13)",
            )
        )

    if false_negatives:
        items.append(
            BacklogItem(
                priority="P0",
                title="Recover missed matches (false negatives) on qualified candidates",
                description=(
                    f"{len(false_negatives)} case(s) where a qualified candidate was "
                    "rejected by the pipeline but a human reviewer would have "
                    "advanced them. False negatives are the costlier failure mode "
                    "for a hiring funnel (a good candidate silently drops out), so "
                    "this should be prioritized above general accuracy tuning."
                ),
                affected_cases=[m.case_id for m in false_negatives],
                recommended_owner="ATS Scoring Engine (Day 13) / Candidate Ranking Engine (Day 14)",
            )
        )

    if rec_only:
        items.append(
            BacklogItem(
                priority="P1",
                title="Align recommendation labels (advance/hold/reject) with human review",
                description=(
                    f"{len(rec_only)} case(s) where the binary shortlist decision "
                    "matched but the finer-grained recommendation label did not "
                    "(e.g. pipeline said 'hold' where a reviewer said 'advance'). "
                    "Suggests the hold/advance threshold in scoring/scorer.py-style "
                    "logic needs recalibration against more labeled examples."
                ),
                affected_cases=[m.case_id for m in rec_only],
                recommended_owner="ATS Scoring Engine (Day 13)",
            )
        )

    # Segment-level signal: flag any role_type/seniority bucket whose
    # accuracy is materially worse than the overall accuracy.
    overall_acc = metrics.overall.accuracy
    weak_segments = []
    for label, counts in {**metrics.by_role_type, **metrics.by_seniority}.items():
        if counts.total and counts.accuracy < overall_acc:
            weak_segments.append((label, counts.accuracy))

    if weak_segments:
        weak_segments.sort(key=lambda x: x[1])
        segment_names = ", ".join(f"{name} ({acc:.0%})" for name, acc in weak_segments)
        items.append(
            BacklogItem(
                priority="P2",
                title="Expand fixture coverage for underperforming segments",
                description=(
                    f"Segments trailing the overall accuracy of {overall_acc:.0%}: "
                    f"{segment_names}. Recommend adding more labeled fixtures in "
                    "these segments before drawing firm conclusions -- current "
                    "sample sizes per segment are small (2-4 cases), so these "
                    "gaps may be noise rather than a systemic pipeline issue."
                ),
                affected_cases=[],
                recommended_owner="QA / Day 17 fixture maintainers",
            )
        )

    if not items:
        items.append(
            BacklogItem(
                priority="P2",
                title="No defects found in current fixture set -- expand coverage",
                description=(
                    "All current fixtures matched human review exactly. This is a "
                    "small, hand-curated set (8 cases); recommend growing the "
                    "fixture library (more roles, edge cases like career changers, "
                    "employment gaps, non-English resumes) before treating this "
                    "as a clean bill of health."
                ),
                affected_cases=[],
                recommended_owner="QA / Day 17 fixture maintainers",
            )
        )

    return items
