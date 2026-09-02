"""
Day 7 metadata envelope.

NOTE: exact Day 7 envelope shape from the wider Zecpath pipeline was
not available to this package at build time. This implements the
common envelope pattern (metadata block + data payload + summary
counts) used elsewhere in the pipeline. If your existing Day 7
envelope module has a different field layout, adjust ENVELOPE
fields below to match — the rest of the engine does not depend on
this shape.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .schema import CandidateEligibilityResult, EligibilityStatus

ENGINE_NAME = "eligibility_decision_engine"
ENGINE_VERSION = "1.0.0"


def build_envelope(
    results: list[CandidateEligibilityResult],
    *,
    source: str = "unknown",
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Wrap eligibility results in a Day 7-style metadata envelope."""
    counts = {status.value: 0 for status in EligibilityStatus}
    for r in results:
        counts[r.status.value] += 1

    metadata = {
        "engine": ENGINE_NAME,
        "engine_version": ENGINE_VERSION,
        "day": "Day 21",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "record_count": len(results),
        "status_counts": counts,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return {
        "metadata": metadata,
        "data": [r.to_dict() for r in results],
    }
