"""
weights.py
----------
Configurable, per-role weight system for the ATS scoring formula.

A WeightProfile assigns a weight to each of the four scoring components
(skill_match, experience_relevance, education_alignment, semantic_similarity)
and must sum to 1.0. WeightProfileRegistry holds a named set of profiles
("default" plus role-specific overrides), resolvable by role name, and can
be loaded from / saved to a JSON config file so recruiters can tune weights
per role without touching code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

COMPONENT_NAMES = (
    "skill_match",
    "experience_relevance",
    "education_alignment",
    "semantic_similarity",
)

_WEIGHT_SUM_TOLERANCE = 0.01


def _normalize_role_key(role: str) -> str:
    return role.strip().lower().replace(" ", "_").replace("-", "_")


@dataclass
class WeightProfile:
    name: str
    weights: Dict[str, float]

    def __post_init__(self) -> None:
        missing = set(COMPONENT_NAMES) - set(self.weights.keys())
        if missing:
            raise ValueError(
                f"Weight profile '{self.name}' is missing components: {sorted(missing)}"
            )
        extra = set(self.weights.keys()) - set(COMPONENT_NAMES)
        if extra:
            raise ValueError(
                f"Weight profile '{self.name}' has unknown components: {sorted(extra)}"
            )
        for name, value in self.weights.items():
            if value < 0:
                raise ValueError(
                    f"Weight profile '{self.name}' has a negative weight for '{name}': {value}"
                )
        total = sum(self.weights.values())
        if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                f"Weight profile '{self.name}' weights must sum to 1.0 (got {total:.4f})"
            )

    def to_dict(self) -> Dict[str, float]:
        return dict(self.weights)


DEFAULT_PROFILES: Dict[str, WeightProfile] = {
    "default": WeightProfile(
        "default",
        {
            "skill_match": 0.35,
            "experience_relevance": 0.25,
            "education_alignment": 0.15,
            "semantic_similarity": 0.25,
        },
    ),
    "software_engineer": WeightProfile(
        "software_engineer",
        {
            "skill_match": 0.40,
            "experience_relevance": 0.30,
            "education_alignment": 0.10,
            "semantic_similarity": 0.20,
        },
    ),
    "data_scientist": WeightProfile(
        "data_scientist",
        {
            "skill_match": 0.35,
            "experience_relevance": 0.20,
            "education_alignment": 0.20,
            "semantic_similarity": 0.25,
        },
    ),
    "sales": WeightProfile(
        "sales",
        {
            "skill_match": 0.25,
            "experience_relevance": 0.40,
            "education_alignment": 0.10,
            "semantic_similarity": 0.25,
        },
    ),
    "executive": WeightProfile(
        "executive",
        {
            "skill_match": 0.20,
            "experience_relevance": 0.45,
            "education_alignment": 0.15,
            "semantic_similarity": 0.20,
        },
    ),
}


class WeightProfileRegistry:
    """Holds named WeightProfiles, resolvable by role, with a mandatory 'default'."""

    def __init__(self, profiles: Optional[Dict[str, WeightProfile]] = None):
        if profiles:
            self._profiles = {_normalize_role_key(k): v for k, v in profiles.items()}
            if "default" not in self._profiles:
                self._profiles["default"] = DEFAULT_PROFILES["default"]
        else:
            self._profiles = dict(DEFAULT_PROFILES)

    def get(self, role: Optional[str]) -> WeightProfile:
        if not role:
            return self._profiles["default"]
        key = _normalize_role_key(role)
        return self._profiles.get(key, self._profiles["default"])

    def register(self, profile: WeightProfile, overwrite: bool = True) -> None:
        key = _normalize_role_key(profile.name)
        if key in self._profiles and not overwrite:
            raise ValueError(f"Weight profile '{key}' already exists")
        self._profiles[key] = profile

    def list_roles(self) -> list[str]:
        return sorted(self._profiles.keys())

    @classmethod
    def from_json_file(cls, path: str | Path) -> "WeightProfileRegistry":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        profiles = {role: WeightProfile(role, weights) for role, weights in data.items()}
        return cls(profiles)

    def save_to_json_file(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {name: profile.to_dict() for name, profile in self._profiles.items()}
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
