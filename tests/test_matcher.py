from ats_engine.matcher import match_resume_to_job


def test_match_returns_score_between_0_and_1(sample_resume_text, sample_job_description):
    result = match_resume_to_job(sample_resume_text, sample_job_description)
    assert 0.0 <= result.score <= 1.0


def test_match_finds_overlapping_keywords(sample_resume_text, sample_job_description):
    result = match_resume_to_job(sample_resume_text, sample_job_description)
    assert "python" in result.matched_keywords
    assert "fastapi" in result.matched_keywords


def test_empty_job_description_returns_zero_score(sample_resume_text):
    result = match_resume_to_job(sample_resume_text, "")
    assert result.score == 0.0
    assert result.is_shortlisted is False


def test_no_overlap_gives_low_score():
    result = match_resume_to_job("Poet and painter.", "Looking for a nuclear physicist.")
    assert result.score < 0.2
