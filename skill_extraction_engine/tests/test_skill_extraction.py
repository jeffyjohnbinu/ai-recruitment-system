"""
Automated test suite for the Skill Extraction Engine (Day 9).

Run with:
    python -m pytest skill_extraction_engine/tests/test_skill_extraction.py -v

Or use tests/run_tests.py to also generate a timestamped log file.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from skill_extraction_engine.extractor import SkillExtractionEngine  # noqa: E402
from skill_extraction_engine.matcher import SkillMatcher  # noqa: E402
from skill_extraction_engine.matcher import SkillMention  # noqa: E402
from skill_extraction_engine.normalizer import normalize_mentions  # noqa: E402
from skill_extraction_engine.skill_dictionary import SKILL_DICTIONARY  # noqa: E402

FIXTURES_DIR = Path(__file__).parent / "fixtures"
OUTPUT_DIR = Path(__file__).parent / "test_outputs"


@pytest.fixture(scope="module")
def engine():
    return SkillExtractionEngine(output_dir=OUTPUT_DIR)


@pytest.fixture(scope="module")
def clean_resume_result(engine):
    return engine.extract_from_file(FIXTURES_DIR / "sample_resume_clean_1.txt")


@pytest.fixture(scope="module")
def noisy_resume_result(engine):
    return engine.extract_from_file(FIXTURES_DIR / "sample_resume_clean_2_noisy.txt")


def _skill_map(result):
    return {r.skill: r for r in result.skills}


# --------------------------------------------------------------------- #
# Exact / alias matching
# --------------------------------------------------------------------- #
def test_exact_technical_skill_detected(clean_resume_result):
    skills = _skill_map(clean_resume_result)
    assert "Python" in skills
    assert skills["Python"].category == "technical"
    assert skills["Python"].match_type == "exact"


def test_alias_match_resolves_to_canonical(noisy_resume_result):
    skills = _skill_map(noisy_resume_result)
    assert "React" in skills, "alias 'Reactjs' should resolve to canonical 'React'"
    assert skills["React"].match_type == "alias"


def test_multiword_alias_matched(noisy_resume_result):
    skills = _skill_map(noisy_resume_result)
    assert "Node.js" in skills, "alias 'Node JS' should resolve to canonical 'Node.js'"


def test_business_skill_detected(clean_resume_result):
    skills = _skill_map(clean_resume_result)
    assert "Agile" in skills
    assert skills["Agile"].category == "business"


def test_creative_skill_detected(noisy_resume_result):
    skills = _skill_map(noisy_resume_result)
    assert "UI/UX Design" in skills
    assert skills["UI/UX Design"].category == "creative"


# --------------------------------------------------------------------- #
# Skill stacks
# --------------------------------------------------------------------- #
def test_stack_expands_into_constituent_skills(noisy_resume_result):
    skills = _skill_map(noisy_resume_result)
    for expected in ["MongoDB", "Express.js", "React", "Node.js"]:
        assert expected in skills, f"MERN stack should expand to include {expected}"


def test_stack_inferred_records_source(noisy_resume_result):
    skills = _skill_map(noisy_resume_result)
    express = skills["Express.js"]
    # Express.js isn't mentioned directly in the noisy fixture -- it can
    # only have been found via the MERN stack expansion.
    assert express.match_type == "stack_inferred"
    assert "MERN" in express.stack_sources


def test_stack_inferred_scores_lower_than_direct_mention(noisy_resume_result):
    skills = _skill_map(noisy_resume_result)
    # React is mentioned directly (alias "Reactjs") AND via the MERN stack;
    # the direct alias match should win and outscore a pure stack-inferred skill.
    assert skills["React"].confidence > skills["Express.js"].confidence


# --------------------------------------------------------------------- #
# Fuzzy / spelling-variation matching
# --------------------------------------------------------------------- #
def test_fuzzy_matches_misspelled_kubernetes(noisy_resume_result):
    skills = _skill_map(noisy_resume_result)
    assert "Kubernetes" in skills, "'Kubernets' should fuzzy-match to Kubernetes"
    assert skills["Kubernetes"].match_type == "fuzzy"


def test_fuzzy_matches_misspelled_python(noisy_resume_result):
    skills = _skill_map(noisy_resume_result)
    assert "Python" in skills, "'Pyth0n' should fuzzy-match to Python"


def test_fuzzy_confidence_capped_below_exact():
    matcher = SkillMatcher()
    text = "Experienced with Kubernets and container orchestration."
    mentions, covered = matcher.match_phrases(text)
    fuzzy_mentions = matcher.match_fuzzy(text, covered)
    assert fuzzy_mentions
    records = normalize_mentions(fuzzy_mentions)
    kube = next(r for r in records if r.skill == "Kubernetes")
    assert kube.confidence < 1.0


def test_fuzzy_does_not_overmatch_short_random_words():
    matcher = SkillMatcher()
    text = "The cat sat on a mat and had a nap in May."
    mentions, covered = matcher.match_phrases(text)
    fuzzy_mentions = matcher.match_fuzzy(text, covered)
    assert fuzzy_mentions == [], "everyday short words should not fuzzy-match real skills"


# --------------------------------------------------------------------- #
# Deduplication / normalization
# --------------------------------------------------------------------- #
def test_repeated_mentions_deduplicated_into_one_record(clean_resume_result):
    skills = _skill_map(clean_resume_result)
    # "Python" appears twice in the fixture (Summary + Experience + Skills)
    assert skills["Python"].mention_count >= 2
    assert len([s for s in clean_resume_result.skills if s.skill == "Python"]) == 1


def test_normalize_mentions_merges_match_types():
    mentions = [
        SkillMention("Python", "fuzzy", "Pyth0n", 0, 6, fuzzy_ratio=88.0),
        SkillMention("Python", "exact", "Python", 20, 26),
    ]
    records = normalize_mentions(mentions)
    assert len(records) == 1
    record = records[0]
    assert record.match_type == "exact"  # exact beats fuzzy when both present
    assert record.mention_count == 2


# --------------------------------------------------------------------- #
# Confidence scoring
# --------------------------------------------------------------------- #
def test_confidence_scores_bounded(clean_resume_result, noisy_resume_result):
    for result in (clean_resume_result, noisy_resume_result):
        for record in result.skills:
            assert 0.0 <= record.confidence <= 1.0


def test_skills_section_gives_confidence_bonus():
    # Use a fuzzy (spelling-variant) match, whose base score sits below the
    # 1.0 ceiling, so the section bonus has headroom to show up. An exact
    # match already scores 1.0 and would saturate regardless of the bonus.
    engine = SkillExtractionEngine(output_dir=OUTPUT_DIR)
    text_with_section = "Summary\nWorked on various projects.\nSkills\nKubernets\n"
    text_without_section = "Summary\nWorked on various projects using Kubernets.\n"

    with_section = engine.extract_from_text(text_with_section, source_name="a", persist=False)
    without_section = engine.extract_from_text(text_without_section, source_name="b", persist=False)

    conf_with = next(r.confidence for r in with_section.skills if r.skill == "Kubernetes")
    conf_without = next(r.confidence for r in without_section.skills if r.skill == "Kubernetes")
    assert conf_with > conf_without


def test_results_sorted_by_confidence_descending(clean_resume_result):
    confidences = [r.confidence for r in clean_resume_result.skills]
    assert confidences == sorted(confidences, reverse=True)


# --------------------------------------------------------------------- #
# Result / status handling
# --------------------------------------------------------------------- #
def test_empty_text_returns_failed_status(engine):
    result = engine.extract_from_text("", source_name="empty", persist=False)
    assert result.status == "failed"
    assert result.total_skills_found == 0


def test_text_with_no_known_skills_returns_partial(engine):
    result = engine.extract_from_text(
        "This paragraph mentions no recognizable skills at all.",
        source_name="no_skills",
        persist=False,
    )
    assert result.status == "partial"
    assert result.total_skills_found == 0
    assert result.warnings


def test_skills_by_category_counts_match_records(clean_resume_result):
    total_from_categories = sum(clean_resume_result.skills_by_category.values())
    assert total_from_categories == clean_resume_result.total_skills_found


def test_output_json_written_to_disk(clean_resume_result):
    expected = OUTPUT_DIR / "structured" / "sample_resume_clean_1.skills.json"
    assert expected.exists()


# --------------------------------------------------------------------- #
# Dictionary sanity
# --------------------------------------------------------------------- #
def test_dictionary_covers_all_three_categories():
    categories = {d.category for d in SKILL_DICTIONARY.values()}
    assert categories == {"technical", "business", "creative"}


def test_no_duplicate_aliases_across_skills():
    seen = {}
    for canonical, definition in SKILL_DICTIONARY.items():
        for alias in [canonical] + definition.aliases:
            key = alias.lower()
            assert (
                key not in seen
            ), f"alias '{alias}' claimed by both {seen.get(key)} and {canonical}"
            seen[key] = canonical


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
