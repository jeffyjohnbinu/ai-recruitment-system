from scoring.scorer import compute_final_score, rank_candidates


def test_final_score_advance_recommendation_boosts_score():
    result = compute_final_score(ats_score=0.5, ai_recommendation="advance")
    assert result.final_score > 0.5


def test_final_score_reject_recommendation_lowers_score():
    result = compute_final_score(ats_score=0.8, ai_recommendation="reject")
    assert result.final_score < 0.8


def test_final_score_bounded_between_0_and_1():
    result = compute_final_score(ats_score=1.0, ai_recommendation="advance")
    assert 0.0 <= result.final_score <= 1.0


def test_rank_candidates_orders_descending():
    scores = [
        compute_final_score(0.3, "hold"),
        compute_final_score(0.9, "advance"),
        compute_final_score(0.1, "reject"),
    ]
    ranked = rank_candidates(scores)
    assert ranked[0].final_score >= ranked[1].final_score >= ranked[2].final_score
