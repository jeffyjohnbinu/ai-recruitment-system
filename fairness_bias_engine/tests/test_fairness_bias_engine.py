"""
Automated test suite for the Fairness, Normalization & Bias Reduction
Engine (Day 15).

Run with:
    python -m pytest fairness_bias_engine/tests/test_fairness_bias_engine.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fairness_bias_engine.bias_auditor import BiasAuditor  # noqa: E402
from fairness_bias_engine.engine import FairnessBiasEngine  # noqa: E402
from fairness_bias_engine.keyword_balancer import KeywordDependencyReducer  # noqa: E402
from fairness_bias_engine.masker import AttributeMasker  # noqa: E402
from fairness_bias_engine.normalizer import ResumeNormalizer  # noqa: E402
from fairness_bias_engine.score_normalizer import ScoreNormalizer  # noqa: E402


# --------------------------------------------------------------------- #
# ResumeNormalizer
# --------------------------------------------------------------------- #
def test_normalizer_canonicalizes_degree_variants():
    normalizer = ResumeNormalizer()
    profile = {"candidate_id": "c1", "highest_degree": "M.Tech in Data Science"}
    canonical = normalizer.normalize(profile)
    assert canonical.education_level == "Masters"


def test_normalizer_flattens_categorized_skills_and_dedupes():
    normalizer = ResumeNormalizer()
    profile = {
        "candidate_id": "c2",
        "skills": {"Languages": ["Python", "python", "SQL"], "Cloud": ["AWS", "  aws  "]},
    }
    canonical = normalizer.normalize(profile)
    assert canonical.standardized_skills == ["aws", "python", "sql"]


def test_normalizer_derives_experience_from_records_when_missing():
    normalizer = ResumeNormalizer()
    profile = {
        "candidate_id": "c3",
        "experience_records": [{"duration_months": 24}, {"duration_months": 12}],
    }
    canonical = normalizer.normalize(profile)
    assert canonical.total_experience_years == 3.0


def test_normalizer_handles_unrecognized_degree_gracefully():
    normalizer = ResumeNormalizer()
    profile = {"candidate_id": "c4", "highest_degree": "Certificate of Excellence"}
    canonical = normalizer.normalize(profile)
    assert canonical.education_level == "Other"
    assert canonical.normalization_notes  # a note should explain the fallback


def test_normalizer_defaults_missing_experience_to_zero():
    normalizer = ResumeNormalizer()
    profile = {"candidate_id": "c5"}
    canonical = normalizer.normalize(profile)
    assert canonical.total_experience_years == 0.0
    assert canonical.education_level == "Unknown"


# --------------------------------------------------------------------- #
# AttributeMasker
# --------------------------------------------------------------------- #
def test_masker_removes_protected_fields():
    masker = AttributeMasker()
    profile = {"name": "Jordan Lee", "gender": "female", "age": 29, "skills": ["python"]}
    result = masker.mask(profile)
    assert result.masked_profile["name"] == "[REDACTED]"
    assert result.masked_profile["gender"] == "[REDACTED]"
    assert result.masked_profile["age"] == "[REDACTED]"
    assert result.masked_profile["skills"] == ["python"]
    assert set(result.removed_fields) == {"name", "gender", "age"}


def test_masker_keeps_job_relevant_fields_untouched():
    masker = AttributeMasker()
    profile = {"name": "Sam Rivera", "total_experience_years": 5, "education_level": "Bachelors"}
    result = masker.mask(profile)
    assert result.masked_profile["total_experience_years"] == 5
    assert result.masked_profile["education_level"] == "Bachelors"


def test_masker_neutralizes_gendered_pronouns_in_free_text():
    masker = AttributeMasker()
    profile = {"summary": "She led the migration project and mentored her team."}
    result = masker.mask(profile)
    text = result.masked_profile["summary"]
    assert "She" not in text and "she" not in text
    assert "her" not in text
    assert "summary" in result.neutralized_text_fields


def test_masker_keeps_city_region_but_drops_street_address():
    masker = AttributeMasker()
    profile = {"address": "123 Main St, Austin, TX"}
    result = masker.mask(profile)
    assert "123 Main St" not in result.masked_profile["address"]
    assert "Austin" in result.masked_profile["address"]


def test_masker_masks_institution_name_by_default():
    masker = AttributeMasker()
    profile = {"academic_profile": {"institution": "Harvard University", "degree": "MBA"}}
    result = masker.mask(profile)
    assert result.masked_profile["academic_profile"]["institution"] == "[INSTITUTION]"
    assert result.institution_masked is True


def test_masker_reversible_map_not_in_default_dict_output():
    masker = AttributeMasker()
    profile = {"name": "Alex Kim"}
    result = masker.mask(profile)
    output = result.to_dict(include_reversible_map=False)
    assert "reversible_map" not in output


# --------------------------------------------------------------------- #
# KeywordDependencyReducer
# --------------------------------------------------------------------- #
def test_keyword_reducer_caps_share_and_redistributes():
    reducer = KeywordDependencyReducer(max_keyword_share=0.3)
    components = {
        "keyword_match": (1.0, 0.6),
        "semantic_similarity": (0.5, 0.3),
        "experience_relevance": (0.4, 0.1),
    }
    report = reducer.rebalance(components)
    assert report.dampening_applied is True
    kw_raw, kw_weight = report.adjusted_components["keyword_match"]
    assert kw_weight == pytest.approx(0.3)
    # redistributed weight should have increased the other two components' weights
    total_weight = sum(w for _, w in report.adjusted_components.values())
    assert total_weight == pytest.approx(1.0, abs=1e-6)


def test_keyword_reducer_no_op_when_already_under_cap():
    reducer = KeywordDependencyReducer(max_keyword_share=0.5)
    components = {"keyword_match": (0.8, 0.3), "semantic_similarity": (0.6, 0.7)}
    report = reducer.rebalance(components)
    assert report.dampening_applied is False
    assert report.adjusted_components == components


def test_keyword_reducer_diminishing_returns_curve():
    reducer = KeywordDependencyReducer(diminishing_threshold=5)
    # below threshold: plain ratio
    low = reducer.apply_diminishing_returns(matched_keyword_count=3, total_keyword_count=10)
    assert low == pytest.approx(0.3)
    # above threshold: dampened, should be less than the naive linear ratio
    naive_ratio = 20 / 25
    dampened = reducer.apply_diminishing_returns(matched_keyword_count=20, total_keyword_count=25)
    assert dampened < naive_ratio


def test_keyword_reducer_handles_missing_component_gracefully():
    reducer = KeywordDependencyReducer()
    components = {"semantic_similarity": (0.7, 0.5), "experience_relevance": (0.6, 0.5)}
    report = reducer.rebalance(components)
    assert report.keyword_component_name is None
    assert report.dampening_applied is False
    assert report.adjusted_components == components


# --------------------------------------------------------------------- #
# ScoreNormalizer
# --------------------------------------------------------------------- #
def test_score_normalizer_computes_zscore_and_percentile():
    normalizer = ScoreNormalizer()
    records = normalizer.normalize_pool({"a": 0.9, "b": 0.5, "c": 0.1})
    by_id = {r.candidate_id: r for r in records}
    assert by_id["a"].percentile == pytest.approx(1.0)
    assert by_id["c"].percentile == pytest.approx(1 / 3, abs=1e-3)
    assert by_id["a"].z_score > by_id["b"].z_score > by_id["c"].z_score


def test_score_normalizer_handles_single_candidate_pool():
    normalizer = ScoreNormalizer()
    records = normalizer.normalize_pool({"only": 0.42})
    assert len(records) == 1
    assert records[0].raw_score == 0.42
    assert records[0].percentile == 0.5


def test_score_normalizer_handles_zero_variance_pool():
    normalizer = ScoreNormalizer()
    records = normalizer.normalize_pool({"a": 0.6, "b": 0.6, "c": 0.6})
    assert all(r.z_score == 0.0 for r in records)


def test_score_normalizer_empty_pool_returns_empty_list():
    normalizer = ScoreNormalizer()
    assert normalizer.normalize_pool({}) == []


# --------------------------------------------------------------------- #
# BiasAuditor
# --------------------------------------------------------------------- #
def test_bias_auditor_flags_four_fifths_violation():
    auditor = BiasAuditor()
    candidates = (
        [{"group": "group_a", "score": 0.8, "shortlisted": True} for _ in range(8)]
        + [{"group": "group_a", "score": 0.8, "shortlisted": False} for _ in range(2)]
        + [{"group": "group_b", "score": 0.5, "shortlisted": True} for _ in range(3)]
        + [{"group": "group_b", "score": 0.5, "shortlisted": False} for _ in range(7)]
    )
    report = auditor.audit_dimension("gender", candidates)
    # group_a selection rate 0.8, group_b selection rate 0.3 -> ratio 0.375 < 0.8
    assert report.flagged is True
    assert report.four_fifths_ratio == pytest.approx(0.375)


def test_bias_auditor_no_flag_when_rates_are_close():
    auditor = BiasAuditor()
    candidates = (
        [{"group": "group_a", "score": 0.7, "shortlisted": True} for _ in range(5)]
        + [{"group": "group_a", "score": 0.7, "shortlisted": False} for _ in range(5)]
        + [{"group": "group_b", "score": 0.68, "shortlisted": True} for _ in range(5)]
        + [{"group": "group_b", "score": 0.68, "shortlisted": False} for _ in range(5)]
    )
    report = auditor.audit_dimension("gender", candidates)
    assert report.flagged is False


def test_bias_auditor_reports_insufficient_data_with_no_groups():
    auditor = BiasAuditor()
    report = auditor.audit_dimension("gender", [])
    assert report.insufficient_data is True
    assert "skipped" in report.narrative.lower()


def test_bias_auditor_flags_low_confidence_with_tiny_groups():
    auditor = BiasAuditor()
    candidates = [
        {"group": "group_a", "score": 0.9, "shortlisted": True},
        {"group": "group_b", "score": 0.2, "shortlisted": False},
    ]
    report = auditor.audit_dimension("gender", candidates)
    assert report.insufficient_data is True
    assert all(g.low_confidence for g in report.group_stats)


# --------------------------------------------------------------------- #
# FairnessBiasEngine (integration)
# --------------------------------------------------------------------- #
def test_engine_end_to_end_pipeline():
    engine = FairnessBiasEngine(max_keyword_share=0.3)
    profiles = [
        {
            "candidate_id": "cand_1",
            "name": "Priya Nair",
            "gender": "female",
            "skills": ["python", "sql"],
            "total_experience_years": 4,
            "highest_degree": "B.Tech",
        },
        {
            "candidate_id": "cand_2",
            "name": "John Smith",
            "gender": "male",
            "skills": ["python", "sql", "aws"],
            "total_experience_years": 6,
            "highest_degree": "M.S.",
        },
    ]
    score_components = {
        "cand_1": {"keyword_match": (0.9, 0.6), "semantic_similarity": (0.5, 0.4)},
        "cand_2": {"keyword_match": (0.95, 0.6), "semantic_similarity": (0.8, 0.4)},
    }
    demographics = {"gender": {"cand_1": "female", "cand_2": "male"}}
    shortlist = {"cand_1": True, "cand_2": True}

    result = engine.process_candidate_pool(
        raw_profiles=profiles,
        score_components=score_components,
        role_id="backend_engineer",
        demographic_groups=demographics,
        shortlist_flags=shortlist,
    )

    assert len(result.per_candidate) == 2
    for c in result.per_candidate:
        assert c.masking.masked_profile["name"] == "[REDACTED]"
        assert c.canonical_profile.education_level in ("Bachelors", "Masters")
        assert c.keyword_balance.dampening_applied is True  # both start above 0.3 cap

    assert len(result.normalized_scores) == 2
    assert len(result.bias_reports) == 1
    assert result.bias_reports[0].dimension == "gender"

    record = result.to_storage_record()
    assert record.candidate_count == 2
    assert record.schema_version
    # reversible identity map must never leak into the storage record
    assert "reversible_map" not in str(record.masked_profiles[0].keys())


def test_engine_handles_candidate_with_no_score_components():
    engine = FairnessBiasEngine()
    profiles = [{"candidate_id": "solo", "skills": ["python"]}]
    result = engine.process_candidate_pool(raw_profiles=profiles, score_components={})
    assert result.per_candidate[0].adjusted_final_score == 0.0
    assert result.per_candidate[0].keyword_balance.keyword_component_name is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
