import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from performance_optimization_engine.entity_refinement import EntityRefiner  # noqa: E402


def test_refine_plain_string_list():
    refiner = EntityRefiner()
    result = refiner.refine(["Python", "AWS", "SQL"])
    names = {e.canonical_name for e in result.entities}
    assert "Python" in names
    assert "AWS" in names
    assert result.input_count == 3


def test_refine_deduplicates_near_variants():
    refiner = EntityRefiner()
    result = refiner.refine(["Node.js", "NodeJS", "node js", "Python"])
    assert result.output_count == 2
    assert result.merged_duplicates >= 2


def test_refine_accepts_dict_wrapping_list():
    refiner = EntityRefiner()
    data = {"skills": [{"name": "Docker", "confidence": 0.9}, {"skill": "docker", "score": 0.8}]}
    result = refiner.refine(data)
    assert result.output_count == 1
    assert result.entities[0].merge_count == 2


def test_refine_alias_tolerant_field_names():
    refiner = EntityRefiner()
    data = [
        {"entity": "Kubernetes", "match_confidence": 0.7},
        {"text": "kubernetes", "confidence": 0.6},
    ]
    result = refiner.refine(data)
    assert result.output_count == 1


def test_refine_drops_low_confidence_noise():
    refiner = EntityRefiner(min_confidence=0.5)
    result = refiner.refine([{"name": "Obscure Tool", "confidence": 0.1}])
    assert result.output_count == 0
    assert result.dropped_low_confidence == 1


def test_refine_boosts_confidence_for_confirmed_duplicates():
    refiner = EntityRefiner()
    result = refiner.refine(
        [{"name": "React", "confidence": 0.6}, {"name": "React", "confidence": 0.6}]
    )
    assert result.entities[0].confidence > 0.6


def test_refine_preserves_short_acronyms_casing():
    refiner = EntityRefiner()
    result = refiner.refine(["AWS"])
    # already-uppercase short acronym should be preserved, not lower/title-cased
    assert result.entities[0].canonical_name == "AWS"


def test_refine_rejects_bad_source_type():
    refiner = EntityRefiner()
    with pytest.raises(TypeError):
        refiner.refine(12345)


def test_refine_raises_on_dict_without_known_key():
    refiner = EntityRefiner()
    with pytest.raises(ValueError):
        refiner.refine({"unrelated_key": []})


def test_refine_ignores_pure_noise_tokens():
    refiner = EntityRefiner()
    result = refiner.refine(["Python", "----", "", "   "])
    assert result.output_count == 1


def test_refine_result_to_dict_serializable():
    refiner = EntityRefiner()
    result = refiner.refine(["Java"])
    d = result.to_dict()
    assert "entities" in d
    assert d["output_count"] == 1
