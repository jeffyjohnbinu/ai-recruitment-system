"""
masker.py
---------
Masks non-essential personal attributes before a candidate profile
reaches any scoring stage, in the spirit of "blind hiring". The idea is
simple: a scorer cannot be biased by information it never sees.

What gets masked (never sent downstream to scoring):
    - name
    - gender / gender-coded pronouns in free text
    - date of birth / age
    - photo indicator
    - marital status
    - nationality / religion
    - full street address (city/region kept, since some roles are
      genuinely location-relevant, e.g. on-site requirements)

What is explicitly NOT masked (job-relevant, needed for a fair
evaluation to even be possible):
    - skills, certifications
    - years of experience, titles, seniority
    - education level (degree tier; the specific institution name is
      masked down to a prestige-neutral tier by default -- see
      `mask_institution_name` -- because school-name based scoring is a
      well-documented pathway for pedigree/class bias)

A reversible mapping is kept ONLY inside the MaskingResult returned to
the caller (e.g. for a recruiter who needs to eventually contact the
person) and is never part of the payload handed to scoring/ranking
components.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

SCHEMA_VERSION = "1.0.0"

# Fields that are dropped/replaced entirely before scoring.
_PROTECTED_FIELDS = {
    "name",
    "full_name",
    "candidate_name",
    "gender",
    "sex",
    "date_of_birth",
    "dob",
    "age",
    "photo",
    "photo_url",
    "marital_status",
    "nationality",
    "religion",
    "ethnicity",
    "race",
}

# Address is partially masked: keep city/region, drop street-level detail.
_ADDRESS_FIELDS = {"address", "street_address", "home_address"}

_PRONOUN_MAP = {
    r"\bhe\b": "they",
    r"\bhim\b": "them",
    r"\bhis\b": "their",
    r"\bshe\b": "they",
    r"\bher\b": "their",
    r"\bhers\b": "theirs",
    r"\bhimself\b": "themself",
    r"\bherself\b": "themself",
    r"\bmr\.\b": "",
    r"\bmrs\.\b": "",
    r"\bms\.\b": "",
    r"\bmx\.\b": "",
}
_PRONOUN_PATTERNS = [(re.compile(p, re.IGNORECASE), r) for p, r in _PRONOUN_MAP.items()]

_PLACEHOLDER = "[REDACTED]"


@dataclass
class MaskingResult:
    masked_profile: Dict[str, Any]
    removed_fields: List[str] = field(default_factory=list)
    neutralized_text_fields: List[str] = field(default_factory=list)
    institution_masked: bool = False
    reversible_map: Dict[str, Any] = field(default_factory=dict)  # internal use only

    def to_dict(self, include_reversible_map: bool = False) -> Dict[str, Any]:
        out = {
            "masked_profile": self.masked_profile,
            "removed_fields": self.removed_fields,
            "neutralized_text_fields": self.neutralized_text_fields,
            "institution_masked": self.institution_masked,
            "schema_version": SCHEMA_VERSION,
        }
        if include_reversible_map:
            out["reversible_map"] = self.reversible_map
        return out


class AttributeMasker:
    """Masks non-essential personal attributes on a raw or canonical profile dict."""

    def __init__(self, mask_institution_names: bool = True, neutralize_pronouns: bool = True):
        self.mask_institution_names = mask_institution_names
        self.neutralize_pronouns = neutralize_pronouns

    def mask(self, profile: Dict[str, Any]) -> MaskingResult:
        masked = dict(profile)  # shallow copy; nested dicts handled explicitly below
        removed: List[str] = []
        neutralized: List[str] = []
        reversible: Dict[str, Any] = {}

        for field_name in list(masked.keys()):
            lower_key = field_name.lower()
            if lower_key in _PROTECTED_FIELDS:
                reversible[field_name] = masked[field_name]
                masked[field_name] = _PLACEHOLDER
                removed.append(field_name)
            elif lower_key in _ADDRESS_FIELDS:
                reversible[field_name] = masked[field_name]
                masked[field_name] = self._city_region_only(masked[field_name])

        # Neutralize gendered pronouns / honorifics in free-text fields.
        if self.neutralize_pronouns:
            for text_field in ("summary", "raw_text", "cleaned_text"):
                if text_field in masked and isinstance(masked[text_field], str):
                    new_text, changed = self._neutralize_text(masked[text_field])
                    if changed:
                        masked[text_field] = new_text
                        neutralized.append(text_field)

        institution_masked = False
        if self.mask_institution_names:
            institution_masked = self._mask_institutions(masked, reversible)

        return MaskingResult(
            masked_profile=masked,
            removed_fields=removed,
            neutralized_text_fields=neutralized,
            institution_masked=institution_masked,
            reversible_map=reversible,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _city_region_only(address: Any) -> str:
        if not isinstance(address, str) or not address.strip():
            return _PLACEHOLDER
        parts = [p.strip() for p in address.split(",") if p.strip()]
        # Keep the last one or two comma-separated components (typically city/region),
        # drop street-level detail (first component[s]).
        if len(parts) <= 1:
            return _PLACEHOLDER
        return ", ".join(parts[-2:]) if len(parts) >= 2 else parts[-1]

    def _neutralize_text(self, text: str) -> tuple[str, bool]:
        changed = False
        new_text = text
        for pattern, repl in _PRONOUN_PATTERNS:
            new_text, n = pattern.subn(repl, new_text)
            if n:
                changed = True
        # collapse any double spaces left behind by honorific removal
        new_text = re.sub(r"\s{2,}", " ", new_text)
        return new_text, changed

    def _mask_institutions(self, masked: Dict[str, Any], reversible: Dict[str, Any]) -> bool:
        """Replace specific institution names with a neutral tier placeholder.

        Institution *prestige* correlates with socioeconomic background more
        than with job performance for most roles, so by default we reduce it
        to a generic marker and rely on `education_level` (degree tier) for
        scoring instead.
        """
        touched = False
        academic = masked.get("academic_profile")
        if isinstance(academic, dict) and academic.get("institution"):
            reversible["academic_profile.institution"] = academic["institution"]
            academic = dict(academic)
            academic["institution"] = "[INSTITUTION]"
            masked["academic_profile"] = academic
            touched = True
        elif isinstance(academic, list):
            new_list = []
            for entry in academic:
                if isinstance(entry, dict) and entry.get("institution"):
                    reversible.setdefault("academic_profile", []).append(entry["institution"])
                    entry = dict(entry)
                    entry["institution"] = "[INSTITUTION]"
                    touched = True
                new_list.append(entry)
            masked["academic_profile"] = new_list
        return touched
