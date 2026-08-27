"""
Automated test suite for the ATS Scoring Engine (Day 13).

Run with:
    python -m pytest ats_scoring_engine/tests/test_scoring_engine.py -v

Or use tests/run_tests.py to also generate a timestamped log file.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ats_scoring_engine.explainability import (  # noqa: E402
    build_component_explanations,
    build_narrative,
)
from ats_scoring_engine.loaders import (  # noqa: E402
    extract_education_alignment,
    extract_experience_relevance,
    extract_semantic_similarity,
    extract_skill_match,
)
from ats_scoring_engine.scoring_engine import ATSScoringEngine  # noqa: E402
from ats_scoring_engine.storage import ResultStore, build_score_record  # noqa: E402
from ats_scoring_engine.weights import WeightProfile, WeightProfileRegistry  # noqa: E402

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture
def skill_data():
    return _load_fixture("skill_extraction_sample.json")


@pytest.fixture
def experience_data():
    return _load_fixture("experience_parsing_sample.json")


@pytest.fixture
def education_data():
    return _load_fixture("education_certification_sample.json")


@pytest.fixture
def semantic_data():
    return _load_fixture("semantic_matching_sample.json")


# --------------------------------------------------------------------- #
# Loaders — direct fields, ratio fallbacks, missing data
# --------------------------------------------------------------------- #
def test_skill_match_direct_field(skill_data):
    result = extract_skill_match(skill_data)
    assert result.available
    assert result.score == pytest.approx(0.833, abs=0.01)


def test_skill_match_ratio_fallback():
    data = {
        "matched_skills": ["python", "sql"],
        "required_skills": ["python", "sql", "java", "aws"],
    }
    result = extract_skill_match(data)
    assert result.available
    assert result.score == pytest.approx(0.5)


def test_skill_match_missing_returns_unavailable():
    result = extract_skill_match(None)
    assert result.available is False
    assert result.score is None
    assert result.notes


def test_skill_match_missing_dict_returns_unavailable():
    result = extract_skill_match({"unrelated_field": 1})
    assert result.available is False


def test_experience_relevance_ratio_fallback(experience_data):
    result = extract_experience_relevance(experience_data)
    assert result.available
    assert result.score == pytest.approx(0.9, abs=0.01)  # 4.5 / 5


def test_experience_relevance_direct_field():
    result = extract_experience_relevance({"relevance_score": 0.72})
    assert result.available
    assert result.score == pytest.approx(0.72)


def test_education_alignment_boolean_composite(education_data):
    result = extract_education_alignment(education_data)
    assert result.available
    # 0.6 (degree met) + 0.4 (field match) + 0.1 cert bonus, clamped to 1.0
    assert result.score == pytest.approx(1.0)


def test_education_alignment_partial_boolean():
    result = extract_education_alignment({"meets_minimum_education": True, "field_match": False})
    assert result.available
    assert result.score == pytest.approx(0.6)


def test_education_alignment_missing():
    result = extract_education_alignment({})
    assert result.available is False


def test_semantic_similarity_direct_field(semantic_data):
    result = extract_semantic_similarity(semantic_data)
    assert result.available
    assert result.score == pytest.approx(0.81)


def test_semantic_similarity_handles_negative_cosine():
    result = extract_semantic_similarity({"cosine_similarity": -0.2})
    assert result.available
    assert 0.0 <= result.score <= 1.0


def test_semantic_similarity_missing():
    result = extract_semantic_similarity(None)
    assert result.available is False


# --------------------------------------------------------------------- #
# Weight profiles
# --------------------------------------------------------------------- #
def test_weight_profile_rejects_bad_sum():
    with pytest.raises(ValueError):
        WeightProfile(
            "broken",
            {
                "skill_match": 0.5,
                "experience_relevance": 0.5,
                "education_alignment": 0.5,
                "semantic_similarity": 0.5,
            },
        )


def test_weight_profile_rejects_missing_component():
    with pytest.raises(ValueError):
        WeightProfile("incomplete", {"skill_match": 1.0})


def test_registry_default_profile_exists():
    registry = WeightProfileRegistry()
    profile = registry.get(None)
    assert profile.name == "default"
    assert sum(profile.weights.values()) == pytest.approx(1.0)


def test_registry_resolves_role_case_insensitively():
    registry = WeightProfileRegistry()
    profile = registry.get("Software Engineer")
    assert profile.name == "software_engineer"


def test_registry_unknown_role_falls_back_to_default():
    registry = WeightProfileRegistry()
    profile = registry.get("underwater_basket_weaver")
    assert profile.name == "default"


def test_registry_register_new_profile():
    registry = WeightProfileRegistry()
    custom = WeightProfile(
        "recruiter",
        {
            "skill_match": 0.3,
            "experience_relevance": 0.3,
            "education_alignment": 0.2,
            "semantic_similarity": 0.2,
        },
    )
    registry.register(custom)
    assert registry.get("recruiter").name == "recruiter"


def test_registry_roundtrips_through_json_file(tmp_path):
    registry = WeightProfileRegistry()
    out_path = tmp_path / "weights.json"
    registry.save_to_json_file(out_path)

    loaded = WeightProfileRegistry.from_json_file(out_path)
    assert loaded.get("software_engineer").weights == registry.get("software_engineer").weights


# --------------------------------------------------------------------- #
# ATSScoringEngine — full data
# --------------------------------------------------------------------- #
def test_scoring_all_data_available(skill_data, experience_data, education_data, semantic_data):
    engine = ATSScoringEngine()
    result = engine.compute_score(
        candidate_id="C001",
        job_id="J001",
        skill_data=skill_data,
        experience_data=experience_data,
        education_data=education_data,
        semantic_data=semantic_data,
    )
    assert result.status == "scored"
    assert 0.0 <= result.final_score <= 1.0
    assert not result.components_missing
    assert len(result.components) == 4


def test_scoring_uses_role_specific_profile(
    skill_data, experience_data, education_data, semantic_data
):
    engine = ATSScoringEngine()
    result = engine.compute_score(
        candidate_id="C001",
        job_id="J001",
        skill_data=skill_data,
        experience_data=experience_data,
        education_data=education_data,
        semantic_data=semantic_data,
        role="software_engineer",
    )
    assert result.role_profile == "software_engineer"


def test_scoring_overrides_bypass_extraction():
    engine = ATSScoringEngine()
    result = engine.compute_score(
        candidate_id="C002",
        job_id="J002",
        score_overrides={
            "skill_match": 0.9,
            "experience_relevance": 0.8,
            "education_alignment": 0.7,
            "semantic_similarity": 0.6,
        },
    )
    assert result.status == "scored"
    assert not result.components_missing
    default_weights = engine.registry.get(None).weights
    expected = sum(
        v * default_weights[k]
        for k, v in {
            "skill_match": 0.9,
            "experience_relevance": 0.8,
            "education_alignment": 0.7,
            "semantic_similarity": 0.6,
        }.items()
    )
    assert result.final_score == pytest.approx(expected, abs=1e-6)


# --------------------------------------------------------------------- #
# Missing-data handling
# --------------------------------------------------------------------- #
def test_renormalize_mode_missing_one_component(skill_data, experience_data, education_data):
    engine = ATSScoringEngine(missing_data_mode="renormalize")
    result = engine.compute_score(
        candidate_id="C001",
        job_id="J001",
        skill_data=skill_data,
        experience_data=experience_data,
        education_data=education_data,
        semantic_data=None,  # missing
    )
    assert result.status == "scored"
    assert result.components_missing == ["semantic_similarity"]
    assert result.weights_renormalized is True

    applied = {c.name: c.weight_applied for c in result.components}
    assert applied["semantic_similarity"] == 0.0
    assert sum(applied.values()) == pytest.approx(1.0, abs=1e-6)


def test_neutral_fill_mode_missing_one_component(skill_data, experience_data, education_data):
    engine = ATSScoringEngine(missing_data_mode="neutral_fill", neutral_fill_value=0.5)
    result = engine.compute_score(
        candidate_id="C001",
        job_id="J001",
        skill_data=skill_data,
        experience_data=experience_data,
        education_data=education_data,
        semantic_data=None,
    )
    assert result.status == "scored"
    missing_component = next(c for c in result.components if c.name == "semantic_similarity")
    assert missing_component.raw_score == pytest.approx(0.5)
    assert missing_component.weight_applied == pytest.approx(0.25)  # default profile weight
    assert result.weights_renormalized is False


def test_strict_mode_within_tolerance_still_scores(skill_data, experience_data, education_data):
    engine = ATSScoringEngine(missing_data_mode="strict", max_missing_for_strict=1)
    result = engine.compute_score(
        candidate_id="C001",
        job_id="J001",
        skill_data=skill_data,
        experience_data=experience_data,
        education_data=education_data,
        semantic_data=None,
    )
    assert result.status == "scored"


def test_strict_mode_exceeding_tolerance_returns_insufficient_data(skill_data):
    engine = ATSScoringEngine(missing_data_mode="strict", max_missing_for_strict=1)
    result = engine.compute_score(
        candidate_id="C001",
        job_id="J001",
        skill_data=skill_data,
        experience_data=None,
        education_data=None,
        semantic_data=None,
    )
    assert result.status == "insufficient_data"
    assert result.final_score is None
    assert len(result.components_missing) == 3


def test_all_data_missing_renormalize_yields_zero_weight():
    engine = ATSScoringEngine(missing_data_mode="renormalize")
    result = engine.compute_score(candidate_id="C003", job_id="J003")
    assert result.status == "scored"
    assert result.final_score == 0.0
    assert len(result.components_missing) == 4


def test_invalid_missing_data_mode_raises():
    with pytest.raises(ValueError):
        ATSScoringEngine(missing_data_mode="not_a_real_mode")


# --------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------- #
def test_narrative_mentions_candidate_and_job(
    skill_data, experience_data, education_data, semantic_data
):
    engine = ATSScoringEngine()
    result = engine.compute_score(
        candidate_id="C001",
        job_id="J001",
        skill_data=skill_data,
        experience_data=experience_data,
        education_data=education_data,
        semantic_data=semantic_data,
    )
    narrative = build_narrative(result)
    assert "C001" in narrative
    assert "J001" in narrative
    assert "%" in narrative


def test_narrative_flags_missing_data(skill_data, experience_data, education_data):
    engine = ATSScoringEngine(missing_data_mode="renormalize")
    result = engine.compute_score(
        candidate_id="C001",
        job_id="J001",
        skill_data=skill_data,
        experience_data=experience_data,
        education_data=education_data,
        semantic_data=None,
    )
    narrative = build_narrative(result)
    assert "unavailable" in narrative
    assert "renormalized" in narrative


def test_narrative_insufficient_data_status(skill_data):
    engine = ATSScoringEngine(missing_data_mode="strict", max_missing_for_strict=1)
    result = engine.compute_score(
        candidate_id="C001",
        job_id="J001",
        skill_data=skill_data,
        experience_data=None,
        education_data=None,
        semantic_data=None,
    )
    narrative = build_narrative(result)
    assert "could not be scored" in narrative


def test_component_explanations_cover_all_components(
    skill_data, experience_data, education_data, semantic_data
):
    engine = ATSScoringEngine()
    result = engine.compute_score(
        candidate_id="C001",
        job_id="J001",
        skill_data=skill_data,
        experience_data=experience_data,
        education_data=education_data,
        semantic_data=semantic_data,
    )
    explanations = build_component_explanations(result)
    assert set(explanations.keys()) == {
        "skill_match",
        "experience_relevance",
        "education_alignment",
        "semantic_similarity",
    }
    for text in explanations.values():
        assert isinstance(text, str) and text


# --------------------------------------------------------------------- #
# Storage — metadata envelope + persistence
# --------------------------------------------------------------------- #
def test_score_record_has_metadata_envelope(
    skill_data, experience_data, education_data, semantic_data
):
    engine = ATSScoringEngine()
    result = engine.compute_score(
        candidate_id="C001",
        job_id="J001",
        skill_data=skill_data,
        experience_data=experience_data,
        education_data=education_data,
        semantic_data=semantic_data,
    )
    record = build_score_record(result)
    assert record.schema_version
    assert record.model_version
    assert record.pipeline_version
    assert record.candidate_id == "C001"
    assert record.job_id == "J001"


def test_result_store_writes_json_file(
    tmp_path, skill_data, experience_data, education_data, semantic_data
):
    engine = ATSScoringEngine()
    result = engine.compute_score(
        candidate_id="C001",
        job_id="J001",
        skill_data=skill_data,
        experience_data=experience_data,
        education_data=education_data,
        semantic_data=semantic_data,
    )
    record = build_score_record(result)
    store = ResultStore(tmp_path)
    out_path = store.save(record)

    saved = json.loads(Path(out_path).read_text(encoding="utf-8"))
    assert saved["candidate_id"] == "C001"
    assert saved["job_id"] == "J001"
    assert "narrative_explanation" in saved
    assert "components" in saved and len(saved["components"]) == 4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
