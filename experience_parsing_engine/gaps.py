"""
Gap & Overlap Detection
------------------------
Given a list of parsed ExperienceEntry objects, this module:

  - Computes TOTAL distinct months of experience by merging overlapping
    date ranges into a union (so overlapping roles aren't double-counted).
  - Detects employment GAPS: stretches of time between roles with no
    coverage, above a configurable threshold (default > 1 month).
  - Detects OVERLAPPING roles: two or more roles whose date ranges
    intersect (e.g. a side project alongside a full-time job, or
    imprecise/overstated resume dates worth flagging for review).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .parser import ExperienceEntry

YearMonth = Tuple[int, int]


def _ym_to_index(ym: YearMonth) -> int:
    return ym[0] * 12 + (ym[1] - 1)


def _index_to_ym(idx: int) -> YearMonth:
    return (idx // 12, (idx % 12) + 1)


def _iso_to_ym(iso: Optional[str]) -> Optional[YearMonth]:
    if not iso:
        return None
    year, month = iso.split("-")
    return (int(year), int(month))


@dataclass
class GapPeriod:
    after_role: str  # title @ company of the role preceding the gap
    before_role: str  # title @ company of the role following the gap
    gap_start: str  # ISO YYYY-MM, first month with no coverage
    gap_end: str  # ISO YYYY-MM, last month with no coverage
    gap_months: int


@dataclass
class OverlapPeriod:
    role_a: str
    role_b: str
    overlap_start: str
    overlap_end: str
    overlap_months: int


@dataclass
class ExperienceTimeline:
    total_months: int
    total_years: float
    gaps: List[GapPeriod]
    overlaps: List[OverlapPeriod]
    valid_entry_count: int
    skipped_entry_count: int


def _role_label(entry: ExperienceEntry) -> str:
    return f"{entry.title} @ {entry.company}"


def analyze_timeline(
    entries: List[ExperienceEntry], gap_threshold_months: int = 1
) -> ExperienceTimeline:
    """
    Build a timeline summary from parsed experience entries.

    Entries with unparseable dates (`date_parse_ok=False`) are excluded
    from gap/overlap/total-month math but counted in `skipped_entry_count`
    so the caller can surface a data-quality warning.
    """
    valid = [e for e in entries if e.date_parse_ok and e.start_date and e.end_date]
    skipped = len(entries) - len(valid)

    if not valid:
        return ExperienceTimeline(
            total_months=0,
            total_years=0.0,
            gaps=[],
            overlaps=[],
            valid_entry_count=0,
            skipped_entry_count=skipped,
        )

    # Sort by start date for gap/overlap sweep.
    ordered = sorted(valid, key=lambda e: _ym_to_index(_iso_to_ym(e.start_date)))

    overlaps: List[OverlapPeriod] = []
    gaps: List[GapPeriod] = []

    # --- Overlap detection: sweep sorted-by-start intervals, compare
    # each entry against all previously opened intervals that haven't
    # yet closed before this one starts. ---
    intervals = [
        (_ym_to_index(_iso_to_ym(e.start_date)), _ym_to_index(_iso_to_ym(e.end_date)), e)
        for e in ordered
    ]
    for i in range(len(intervals)):
        a_start, a_end, a_entry = intervals[i]
        for j in range(i + 1, len(intervals)):
            b_start, b_end, b_entry = intervals[j]
            if b_start > a_end:
                break  # sorted by start -> no further j can overlap with a
            overlap_start_idx = max(a_start, b_start)
            overlap_end_idx = min(a_end, b_end)
            if overlap_start_idx <= overlap_end_idx:
                overlaps.append(
                    OverlapPeriod(
                        role_a=_role_label(a_entry),
                        role_b=_role_label(b_entry),
                        overlap_start=_ym_iso(overlap_start_idx),
                        overlap_end=_ym_iso(overlap_end_idx),
                        overlap_months=overlap_end_idx - overlap_start_idx + 1,
                    )
                )

    # --- Total distinct months: merge intervals into a union. ---
    merged: List[List[int]] = []
    for start_idx, end_idx, _ in sorted(intervals, key=lambda t: t[0]):
        if merged and start_idx <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end_idx)
        else:
            merged.append([start_idx, end_idx])

    total_months = sum(end - start + 1 for start, end in merged)

    # --- Gap detection: look at the space between consecutive merged
    # (union) blocks — this correctly ignores overlapping roles instead
    # of flagging false gaps caused by unsorted individual entries. ---
    for k in range(len(merged) - 1):
        prev_end = merged[k][1]
        next_start = merged[k + 1][0]
        gap_len = next_start - prev_end - 1
        if gap_len >= gap_threshold_months:
            # Attribute the gap to the entries whose end/start define
            # the boundary of each merged block.
            after_entry = max(
                (e for e in ordered if _ym_to_index(_iso_to_ym(e.end_date)) == prev_end),
                key=lambda e: _ym_to_index(_iso_to_ym(e.start_date)),
            )
            before_entry = min(
                (e for e in ordered if _ym_to_index(_iso_to_ym(e.start_date)) == next_start),
                key=lambda e: _ym_to_index(_iso_to_ym(e.end_date)),
            )
            gaps.append(
                GapPeriod(
                    after_role=_role_label(after_entry),
                    before_role=_role_label(before_entry),
                    gap_start=_ym_iso(prev_end + 1),
                    gap_end=_ym_iso(next_start - 1),
                    gap_months=gap_len,
                )
            )

    return ExperienceTimeline(
        total_months=total_months,
        total_years=round(total_months / 12, 2),
        gaps=gaps,
        overlaps=overlaps,
        valid_entry_count=len(valid),
        skipped_entry_count=skipped,
    )


def _ym_iso(idx: int) -> str:
    y, m = _index_to_ym(idx)
    return f"{y:04d}-{m:02d}"
