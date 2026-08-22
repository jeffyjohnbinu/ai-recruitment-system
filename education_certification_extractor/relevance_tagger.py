"""
relevance_tagger.py
---------------------
Tags a certification name with a relevance category, so downstream
matching/scoring modules can quickly filter "does this candidate have a
Cloud certification" without re-parsing free text.

Keyword-lookup approach (fast, deterministic, no model dependency) mirrors
the two-pass philosophy used elsewhere in this project: an unmatched
certification isn't dropped, it's tagged "Other" and still returned.
"""

from __future__ import annotations

import re
from typing import List, Tuple

# Ordered (most specific first) so e.g. "aws certified security" hits
# Security before the generic "aws" Cloud rule would otherwise win.
_CATEGORY_KEYWORDS: List[Tuple[str, List[str]]] = [
    (
        "Security",
        [
            "cissp",
            "ceh",
            "comptia security",
            "security+",
            "oscp",
            "cism",
            "cisa",
            "gsec",
            "certified ethical hacker",
            "certified information security",
        ],
    ),
    (
        "Cloud",
        [
            "aws certified",
            "amazon web services",
            "microsoft azure",
            "azure certified",
            "google cloud",
            "gcp certified",
            "cloud practitioner",
            "solutions architect",
            "certified kubernetes",
            "cka",
            "ckad",
        ],
    ),
    (
        "Data & AI",
        [
            "data science",
            "machine learning",
            "deep learning",
            "tensorflow",
            "data analyst",
            "data engineer",
            "certified analytics",
            "microsoft certified: azure data",
            "sas certified",
            "tableau",
        ],
    ),
    (
        "Networking & Systems",
        [
            "ccna",
            "ccnp",
            "ccie",
            "comptia network",
            "network+",
            "juniper",
            "linux+",
            "rhce",
            "red hat certified",
        ],
    ),
    (
        "Project Management",
        [
            "pmp",
            "project management professional",
            "prince2",
            "capm",
            "certified associate in project management",
        ],
    ),
    (
        "Agile & Scrum",
        [
            "scrum master",
            "csm",
            "psm",
            "safe agilist",
            "safe scrum",
            "certified scrum",
            "kanban",
        ],
    ),
    (
        "Quality & Process",
        ["six sigma", "lean six sigma", "iso 9001", "cmmi"],
    ),
    (
        "Finance & Business",
        ["cfa", "cpa", "frm", "cma", "chartered financial analyst"],
    ),
    (
        "Programming & Development",
        [
            "oracle certified",
            "java certified",
            "microsoft certified: developer",
            "certified developer",
        ],
    ),
]

_COMPILED = [
    (cat, [re.compile(re.escape(k), re.IGNORECASE) for k in kws]) for cat, kws in _CATEGORY_KEYWORDS
]


def tag_relevance(certification_name: str) -> str:
    """
    Return the best-matching relevance category for a certification name,
    or "Other" if nothing in the keyword table matches.
    """
    if not certification_name:
        return "Other"
    for category, patterns in _COMPILED:
        if any(p.search(certification_name) for p in patterns):
            return category
    return "Other"
