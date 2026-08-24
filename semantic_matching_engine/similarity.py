"""
Similarity Scoring
-------------------
Given two pieces of text (a resume section and a job-description section),
compute a semantic similarity score in [0, 1] using whatever Embedder was
selected by `embeddings.get_embedder()`.

Also provides section-level comparison helpers for the three areas called
out in the Day 12 brief: Skills, Experience summaries, and Project
descriptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .embeddings import Embedder


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors, safe against zero vectors
    (returns 0.0 instead of raising/NaN when either vector has no signal --
    e.g. a section that was empty or entirely out-of-vocabulary)."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    score = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    # Guard against tiny floating point overshoot (e.g. 1.0000000002)
    return max(0.0, min(1.0, score))


@dataclass
class SectionComparison:
    section: str
    resume_text: str
    job_text: str
    similarity: float
    weight: float
    empty: bool = False


@dataclass
class SimilarityBreakdown:
    section_scores: Dict[str, SectionComparison] = field(default_factory=dict)
    overall_score: float = 0.0
    engine_used: str = ""

    def as_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 4),
            "engine_used": self.engine_used,
            "sections": {
                name: {
                    "similarity": round(cmp.similarity, 4),
                    "weight": cmp.weight,
                    "empty": cmp.empty,
                }
                for name, cmp in self.section_scores.items()
            },
        }


# Default relative importance of each section when combining into one
# overall score. Kept configurable via `weights=` on compute_similarity()
# because different job families (e.g. individual-contributor engineering
# vs. people-management roles) reasonably want different emphasis.
DEFAULT_SECTION_WEIGHTS: Dict[str, float] = {
    "skills": 0.45,
    "experience": 0.35,
    "projects": 0.20,
}


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Section weights must sum to a positive number.")
    return {k: v / total for k, v in weights.items()}


def compute_similarity(
    resume_sections: Dict[str, str],
    job_sections: Dict[str, str],
    embedder: Embedder,
    weights: Optional[Dict[str, float]] = None,
) -> SimilarityBreakdown:
    """
    Compute a weighted semantic similarity score between a resume and a job
    description, broken down by section (skills / experience / projects by
    default, but any matching key set works).

    Sections missing or blank on EITHER side are scored 0.0 for that
    section and excluded from weight re-normalization is NOT performed --
    a resume with no "projects" section legitimately scores lower on that
    axis rather than having its other sections inflated to compensate,
    since project experience genuinely may be a job requirement.
    """
    weights = _normalize_weights(weights or DEFAULT_SECTION_WEIGHTS)

    section_names = list(weights.keys())
    resume_texts = [resume_sections.get(name, "") or "" for name in section_names]
    job_texts = [job_sections.get(name, "") or "" for name in section_names]

    # Batch-embed everything in one call per side for efficiency.
    all_texts = resume_texts + job_texts
    non_empty_indices = [i for i, t in enumerate(all_texts) if t.strip()]

    vectors = np.zeros((len(all_texts), 1))
    if non_empty_indices:
        embedded = embedder.embed([all_texts[i] for i in non_empty_indices])
        vectors = np.zeros((len(all_texts), embedded.shape[1]))
        for pos, idx in enumerate(non_empty_indices):
            vectors[idx] = embedded[pos]

    n = len(section_names)
    breakdown = SimilarityBreakdown(engine_used=getattr(embedder, "engine_name", "unknown"))

    weighted_sum = 0.0
    for i, name in enumerate(section_names):
        resume_vec = vectors[i]
        job_vec = vectors[n + i]
        is_empty = (not resume_texts[i].strip()) or (not job_texts[i].strip())
        sim = 0.0 if is_empty else cosine_similarity(resume_vec, job_vec)

        breakdown.section_scores[name] = SectionComparison(
            section=name,
            resume_text=resume_texts[i],
            job_text=job_texts[i],
            similarity=sim,
            weight=weights[name],
            empty=is_empty,
        )
        weighted_sum += sim * weights[name]

    breakdown.overall_score = round(weighted_sum, 6)
    return breakdown
