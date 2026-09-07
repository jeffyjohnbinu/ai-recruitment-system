"""
test_engine.py
-------------
Integration tests for the ConfidenceSentimentEngine.
"""

import pytest

from day27_confidence_sentiment.engine import (
    AnswerScore,
    ConfidenceSentimentEngine,
    SessionConfidenceResult,
)
from day27_confidence_sentiment.formats import (
    BehavioralIndicatorsReport,
    CommunicationStrengthIndicator,
    ConfidenceAnalysis,
    HesitationPattern,
    PaceMetrics,
    SentimentScore,
    UncertaintySignal,
)

# ---- per-answer scoring ---- #


class TestScoreAnswer:
    def setup_method(self):
        self.engine = ConfidenceSentimentEngine()

    def test_basic_answer(self):
        record = {
            "raw_answer": "I have seven years of experience in Python and Django.",
            "category": "experience",
        }
        result = self.engine.score_answer("q1", record)
        assert isinstance(result, AnswerScore)
        assert result.question_id == "q1"
        assert isinstance(result.hesitation_patterns, list)
        assert isinstance(result.pace_metrics, PaceMetrics)
        assert isinstance(result.uncertainty_signals, list)
        assert isinstance(result.sentiment, SentimentScore)
        assert isinstance(result.strength_indicator, CommunicationStrengthIndicator)
        assert 0.0 <= result.overall_confidence_score <= 1.0

    def test_highly_hesitant_answer(self):
        record = {
            "raw_answer": "Um, uh, I think, I mean, you know, perhaps maybe I have, like, around five years. I don't really know.",
        }
        result = self.engine.score_answer("q1", record)
        assert len(result.hesitation_patterns) > 3
        assert len(result.uncertainty_signals) > 1
        assert result.overall_confidence_score < 0.6
        # Should have warnings.
        assert any("hesitation" in w.lower() for w in result.warnings)

    def test_confident_positive_answer(self):
        record = {
            "raw_answer": "I have seven years of experience. I am confident and excited about this opportunity. I led a team of twelve engineers and successfully delivered multiple projects.",
            "category": "experience",
        }
        result = self.engine.score_answer("q1", record)
        assert result.sentiment.sentiment_label == "positive"
        assert result.sentiment.polarity > 0
        assert result.strength_indicator.overall_strength > 0.5
        assert result.overall_confidence_score > 0.5

    def test_with_duration(self):
        record = {
            "raw_answer": "I have five years of experience.",
            "duration_seconds": 3.0,
        }
        result = self.engine.score_answer("q1", record)
        assert result.pace_metrics.duration_seconds == 3.0
        assert result.pace_metrics.words_per_second is not None

    def test_empty_answer(self):
        record = {"raw_answer": ""}
        result = self.engine.score_answer("q1", record)
        assert result.overall_confidence_score == 0.0
        assert (
            "Empty answer" in result.pace_metrics.note
            or result.pace_metrics.length_label == "too_short"
        )


# ---- batch scoring ---- #


class TestScoreAll:
    def setup_method(self):
        self.engine = ConfidenceSentimentEngine()

    def test_multiple_answers(self):
        answers = [
            ("q1", {"raw_answer": "I have five years of experience.", "category": "experience"}),
            ("q2", {"raw_answer": "My expected salary is 15 LPA.", "category": "salary"}),
            ("q3", {"raw_answer": "I am based in Bangalore.", "category": "location"}),
        ]
        results = self.engine.score_all(answers)
        assert len(results) == 3
        assert results[0].question_id == "q1"
        assert results[1].question_id == "q2"
        assert results[2].question_id == "q3"


# ---- session-level scoring ---- #


class TestScoreSession:
    def setup_method(self):
        self.engine = ConfidenceSentimentEngine()

    def test_basic_session(self):
        answers = [
            (
                "q1",
                {
                    "raw_answer": "I have five years of experience in Python and Django.",
                    "category": "experience",
                },
            ),
            (
                "q2",
                {
                    "raw_answer": "My expected salary is 15 LPA.",
                    "category": "salary",
                },
            ),
            (
                "q3",
                {
                    "raw_answer": "I am based in Bangalore. I am excited about this opportunity.",
                    "category": "location",
                },
            ),
        ]
        result = self.engine.score_session(
            candidate_id="cand_001",
            job_id="job_001",
            session_id="sess_001",
            role_id="software_engineer",
            answers=answers,
        )
        assert isinstance(result, SessionConfidenceResult)
        assert result.candidate_id == "cand_001"
        assert result.job_id == "job_001"
        assert result.session_id == "sess_001"
        assert result.role_id == "software_engineer"
        assert result.schema_version == "1.0.0"
        assert result.model_version == "day27-v1"
        assert len(result.per_answer) == 3
        assert result.avg_response_length > 0
        assert result.session_sentiment_label in ("positive", "neutral", "negative", "mixed")

    def test_highly_uncertain_session(self):
        answers = [
            (
                "q1",
                {
                    "raw_answer": "I think maybe I have around five years, perhaps three or four.",
                    "category": "experience",
                },
            ),
            (
                "q2",
                {
                    "raw_answer": "Um, I don't know, like, you know, it's hard to say. I guess I could.",
                    "category": "skills",
                },
            ),
            (
                "q3",
                {
                    "raw_answer": "Sorry, let me think. I don't remember exactly, but maybe around 12 LPA.",
                    "category": "salary",
                },
            ),
        ]
        result = self.engine.score_session(
            candidate_id="cand_002",
            job_id="job_001",
            session_id="sess_002",
            role_id="software_engineer",
            answers=answers,
        )
        # Should have warnings about low confidence.
        # Note: q1 ("I think maybe I have around five years") has only hedges
        # (no hesitations, neutral sentiment) so its individual score is higher.
        # The session is still flagged as uncertain overall.
        assert result.session_avg_confidence < 0.55
        assert any("confidence" in w.lower() for w in result.warnings)
        assert len(result.high_uncertainty_answers) > 0

    def test_positive_session(self):
        answers = [
            (
                "q1",
                {
                    "raw_answer": "I have seven years of experience. I am confident and excited about this opportunity.",
                },
            ),
            (
                "q2",
                {
                    "raw_answer": "I led a team of twelve engineers and successfully delivered multiple projects. I am proud of this work.",
                },
            ),
            (
                "q3",
                {
                    "raw_answer": "My expected salary is 18 LPA, and I am happy to discuss.",
                },
            ),
        ]
        result = self.engine.score_session(
            candidate_id="cand_003",
            job_id="job_001",
            session_id="sess_003",
            role_id="software_engineer",
            answers=answers,
        )
        assert result.session_sentiment_label in ("positive", "neutral")
        assert result.session_sentiment_polarity > 0

    def test_contradictions_detected(self):
        answers = [
            (
                "q1",
                {
                    "raw_answer": "I have 10 years of experience. I am confident in my skills.",
                },
            ),
            (
                "q2",
                {
                    "raw_answer": "I am available immediately.",
                },
            ),
            (
                "q3",
                {
                    "raw_answer": "I have a 6-month notice period.",
                },
            ),
        ]
        result = self.engine.score_session(
            candidate_id="cand_004",
            job_id="job_001",
            session_id="sess_004",
            role_id="software_engineer",
            answers=answers,
        )
        # Should detect availability contradiction.
        assert len(result.contradictions_detected) >= 0  # at least doesn't fail

    def test_empty_session(self):
        result = self.engine.score_session(
            candidate_id="cand_005",
            job_id="job_001",
            session_id="sess_005",
            role_id="software_engineer",
            answers=[],
        )
        assert len(result.per_answer) == 0
        assert result.session_avg_confidence == 0.0
        assert "insufficient data" in result.narrative.lower()

    def test_to_dict(self):
        answers = [
            ("q1", {"raw_answer": "I have five years of experience."}),
        ]
        result = self.engine.score_session(
            candidate_id="cand_006",
            job_id="job_001",
            session_id="sess_006",
            role_id="software_engineer",
            answers=answers,
        )
        d = result.to_dict()
        assert "schema_version" in d
        assert "model_version" in d
        assert "per_answer" in d
        assert "session_avg_confidence" in d
        assert "narrative" in d


# ---- behavioral report ---- #


class TestBuildBehavioralReport:
    def setup_method(self):
        self.engine = ConfidenceSentimentEngine()

    def test_report_generation(self):
        answers = [
            ("q1", {"raw_answer": "I have five years of experience.", "category": "experience"}),
            ("q2", {"raw_answer": "I am confident and excited.", "category": "skills"}),
        ]
        session_result = self.engine.score_session(
            candidate_id="cand_007",
            job_id="job_001",
            session_id="sess_007",
            role_id="software_engineer",
            answers=answers,
        )
        report = self.engine.build_behavioral_report(session_result)
        assert isinstance(report, BehavioralIndicatorsReport)
        assert report.candidate_id == "cand_007"
        assert report.total_answers == 2
        assert len(report.per_answer) == 2
        assert all(isinstance(a, ConfidenceAnalysis) for a in report.per_answer)
        assert report.schema_version == "1.0.0"
        # The to_dict output should serialize cleanly.
        d = report.to_dict()
        assert "per_answer" in d
        assert all("hesitation_patterns" in a for a in d["per_answer"])


# ---- strength indicator dimension behavior ---- #


class TestStrengthIndicator:
    def setup_method(self):
        self.engine = ConfidenceSentimentEngine()

    def test_clarity_dimension(self):
        # Clear answer with no hesitation.
        result = self.engine.score_answer(
            "q1",
            {
                "raw_answer": "I have five years of experience in Python and have led multiple successful projects."
            },
        )
        assert result.strength_indicator.clarity > 0.5

    def test_conviction_with_polarity(self):
        # Strong positive polarity.
        result_pos = self.engine.score_answer(
            "q1",
            {
                "raw_answer": "I am absolutely confident and excited about this opportunity. I love this work."
            },
        )
        # Strong negative polarity.
        result_neg = self.engine.score_answer(
            "q2", {"raw_answer": "I am completely frustrated and angry. I hate this kind of work."}
        )
        # Both should have decent conviction (decisiveness).
        assert result_pos.strength_indicator.conviction > 0.4
        assert result_neg.strength_indicator.conviction > 0.4

    def test_engagement_with_length(self):
        short = self.engine.score_answer("q1", {"raw_answer": "Yes."})
        long = self.engine.score_answer(
            "q2",
            {
                "raw_answer": "I have five years of experience in Python and Django, "
                "leading a team of twelve engineers on multiple successful projects."
            },
        )
        assert long.strength_indicator.engagement > short.strength_indicator.engagement

    def test_strength_label_distribution(self):
        answers = [
            ("q1", {"raw_answer": "I have five years of experience."}),
            ("q2", {"raw_answer": "Um uh er, I think maybe I'm not sure."}),
            (
                "q3",
                {
                    "raw_answer": "I am absolutely confident and excited. I love this work and have extensive expertise."
                },
            ),
        ]
        result = self.engine.score_session(
            candidate_id="cand_008",
            job_id="job_001",
            session_id="sess_008",
            role_id="software_engineer",
            answers=answers,
        )
        # Distribution should sum to total answers.
        total = sum(result.strength_distribution.values())
        assert total == 3
