import json

import pytest

from eligibility_decision_engine.loaders import load_ats_results, load_job_rules


@pytest.fixture
def ats_file(tmp_path):
    data = {
        "candidates": [
            {
                "candidate_id": "C1",
                "job_role": "backend_engineer",
                "ats_score": 88,
                "skills": ["python"],
                "experience_years": 4,
                "location": "Bengaluru",
                "remote_ok": False,
                "availability": "immediate",
            },
            {
                # alt key naming (camelCase / alias variants)
                "candidateId": "C2",
                "jobRole": "backend_engineer",
                "matchScore": 71,
                "skills": ["python", "flask"],
                "experienceYears": 2,
                "location": "Remote",
                "remoteOk": True,
                "availability": "2_weeks",
            },
        ]
    }
    p = tmp_path / "ats.json"
    p.write_text(json.dumps(data))
    return p


@pytest.fixture
def rules_file(tmp_path):
    data = {
        "rules": [
            {
                "job_role": "backend_engineer",
                "min_ats_score": 75,
                "review_band": 10,
                "mandatory_skills": ["python"],
                "min_experience_years": 2,
                "allowed_locations": ["Bengaluru"],
                "allow_remote": True,
            }
        ]
    }
    p = tmp_path / "rules.json"
    p.write_text(json.dumps(data))
    return p


def test_load_ats_results_handles_wrapper_and_aliases(ats_file):
    candidates = load_ats_results(ats_file)
    assert len(candidates) == 2
    assert candidates[0].candidate_id == "C1"
    assert candidates[1].candidate_id == "C2"  # from candidateId
    assert candidates[1].ats_score == 71  # from matchScore
    assert candidates[1].experience_years == 2  # from experienceYears
    assert candidates[1].remote_ok is True  # from remoteOk


def test_load_ats_results_plain_list(tmp_path):
    data = [{"candidate_id": "C1", "job_role": "x", "ats_score": 50}]
    p = tmp_path / "ats.json"
    p.write_text(json.dumps(data))
    candidates = load_ats_results(p)
    assert len(candidates) == 1


def test_load_ats_results_bad_dict_shape(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"nope": []}))
    with pytest.raises(ValueError):
        load_ats_results(p)


def test_load_job_rules_keys_by_job_role(rules_file):
    rules = load_job_rules(rules_file)
    assert "backend_engineer" in rules
    r = rules["backend_engineer"]
    assert r.min_ats_score == 75
    assert r.mandatory_skills == ["python"]


def test_load_job_rules_single_object(tmp_path):
    data = {"job_role": "qa_engineer", "min_ats_score": 60}
    p = tmp_path / "rule.json"
    p.write_text(json.dumps(data))
    rules = load_job_rules(p)
    assert "qa_engineer" in rules
    assert rules["qa_engineer"].min_ats_score == 60
