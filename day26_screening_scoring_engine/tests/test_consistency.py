"""
test_consistency.py
-------------------
Unit tests for the consistency scorer.
"""

from __future__ import annotations

import pytest

from day26_screening_scoring_engine.consistency import score_consistency


class TestConsistency:
    def test_empty_answers_returns_full_score(self):
        result = score_consistency([])
        assert result.score == 1.0

    def test_single_answer_returns_full_score(self):
        result = score_consistency(
            [
                (
                    "Q1",
                    {
                        "category": "experience",
                        "scoring_weight": 5,
                        "raw_answer": "5 years",
                        "extracted": {},
                    },
                )
            ]
        )
        assert result.score == 1.0

    def test_experience_sum_exceeds_total_flags_penalty(self):
        answers = [
            (
                "Q1",
                {
                    "category": "experience",
                    "scoring_weight": 4,
                    "raw_answer": "I have 8 years total experience.",
                    "extracted": {"years_experience": 8.0},
                },
            ),
            (
                "Q2",
                {
                    "category": "experience",
                    "scoring_weight": 5,
                    "raw_answer": "I have 10 years in Python specifically.",
                    "extracted": {"years_experience": 10.0},
                },
            ),
        ]
        result = score_consistency(answers)
        assert result.score < 1.0
        assert "exceeds" in " ".join(result.notes).lower()

    def test_consistent_experience_returns_full_score(self):
        answers = [
            (
                "Q1",
                {
                    "category": "experience",
                    "scoring_weight": 4,
                    "raw_answer": "I have 5 years total experience.",
                    "extracted": {"years_experience": 5.0},
                },
            ),
            (
                "Q2",
                {
                    "category": "experience",
                    "scoring_weight": 3,
                    "raw_answer": "About 4 years in backend.",
                    "extracted": {},
                },
            ),
        ]
        result = score_consistency(answers)
        assert result.score >= 0.9

    def test_salary_disagreement_flags_penalty(self):
        answers = [
            (
                "Q1",
                {
                    "category": "salary",
                    "scoring_weight": 4,
                    "raw_answer": "I expect 5 LPA.",
                    "extracted": {"expected_salary_lakhs": 5.0},
                },
            ),
            (
                "Q2",
                {
                    "category": "salary",
                    "scoring_weight": 4,
                    "raw_answer": "I expect 20 LPA.",
                    "extracted": {"expected_salary_lakhs": 20.0},
                },
            ),
        ]
        result = score_consistency(answers)
        assert result.score < 1.0

    def test_notice_over_one_year_flagged(self):
        answers = [
            (
                "Q1",
                {
                    "category": "notice_period",
                    "scoring_weight": 4,
                    "raw_answer": "My notice period is 500 days.",
                    "extracted": {"notice_period_days": 500},
                },
            ),
        ]
        result = score_consistency(answers)
        assert result.score < 1.0
        assert any(">1 year" in n for n in result.notes)
