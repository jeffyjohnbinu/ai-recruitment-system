"""
normalizer.py
--------------
Collapses raw SkillMention hits (possibly many per skill, from different
match passes and different resume sections) into one deduplicated,
confidence-scored SkillRecord per canonical skill.

Match-type precedence when the same skill was found multiple ways
(e.g. once exact, once fuzzy): exact > alias > stack_inferred > fuzzy.
The *strongest* match type present drives the confidence base score;
every mention still counts toward the frequency bonus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

from .confidence import score_skill
from .matcher import SkillMention
from .skill_dictionary import SKILL_DICTIONARY

_MATCH_TYPE_RANK = {"exact": 3, "alias": 2, "stack_inferred": 1, "fuzzy": 0}


@dataclass
class SkillRecord:
    skill: str
    category: str
    subcategory: str
    confidence: float
    match_type: str
    mention_count: int
    matched_variants: List[str]
    source_sections: List[str]
    stack_sources: List[str] = field(default_factory=list)


def normalize_mentions(
    mentions: List[SkillMention],
    span_sections: Dict[int, str] | None = None,
) -> List[SkillRecord]:
    """
    Group raw mentions by canonical skill, resolve the strongest match
    type, score confidence, and return sorted (highest confidence first,
    then alphabetically) SkillRecord list.

    `span_sections` optionally maps a mention's `start` offset to the
    resume section name it fell inside (see extractor.py), used only for
    the section-bearing confidence bonus.
    """
    span_sections = span_sections or {}

    grouped: Dict[str, List[SkillMention]] = {}
    for mention in mentions:
        grouped.setdefault(mention.canonical, []).append(mention)

    records: List[SkillRecord] = []
    for canonical, group in grouped.items():
        definition = SKILL_DICTIONARY.get(canonical)
        if definition is None:
            # Should not happen (stack constituents are always dictionary
            # entries), but fail safe rather than crash the pipeline.
            category, subcategory = "unknown", "unknown"
        else:
            category, subcategory = definition.category, definition.subcategory

        best = max(group, key=lambda m: _MATCH_TYPE_RANK.get(m.match_type, -1))
        mention_count = len(group)

        variants = sorted({m.matched_text for m in group}, key=str.lower)
        source_sections: Set[str] = {
            span_sections[m.start] for m in group if m.start in span_sections
        }
        stack_sources = sorted({m.stack_source for m in group if m.stack_source})

        fuzzy_ratio = best.fuzzy_ratio if best.match_type == "fuzzy" else None
        confidence = score_skill(
            match_type=best.match_type,
            mention_count=mention_count,
            source_sections=source_sections,
            fuzzy_ratio=fuzzy_ratio,
        )

        records.append(
            SkillRecord(
                skill=canonical,
                category=category,
                subcategory=subcategory,
                confidence=confidence,
                match_type=best.match_type,
                mention_count=mention_count,
                matched_variants=variants,
                source_sections=sorted(source_sections),
                stack_sources=stack_sources,
            )
        )

    records.sort(key=lambda r: (-r.confidence, r.skill.lower()))
    return records
