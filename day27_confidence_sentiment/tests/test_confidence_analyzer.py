"""
test_confidence_analyzer.py
--------------------------
Unit tests for confidence_analyzer.py
"""

import pytest

from day27_confidence_sentiment.confidence_analyzer import (
    HesitationPattern,
    PaceMetrics,
    UncertaintySignal,
    analyze_confidence,
    detect_hesitation_patterns,
    detect_uncertainty,
    measure_pace,
)

# ---- hesitation detection ---- #


class TestDetectHesitationPatterns:
    def test_filler_words(self):
        text = "Um, I think uh, I have like five years of experience um and I'm very confident."
        patterns = detect_hesitation_patterns(text)
        types = [p.pattern_type for p in patterns]
        assert "filler" in types
        filler_count = sum(1 for p in patterns if p.pattern_type == "filler")
        assert filler_count >= 2

    def test_repetition(self):
        text = "I I I have worked at multiple companies companies."
        patterns = detect_hesitation_patterns(text)
        types = [p.pattern_type for p in patterns]
        assert "repetition" in types

    def test_stutter(self):
        text = "I've been w-w-working in this field for five years."
        patterns = detect_hesitation_patterns(text)
        assert len(patterns) > 0
        types = [p.pattern_type for p in patterns]
        assert "repetition" in types

    def test_repair_i_mean(self):
        text = "I have five years, I mean, actually six years of experience."
        patterns = detect_hesitation_patterns(text)
        types = [p.pattern_type for p in patterns]
        assert "repair" in types

    def test_repair_sorry(self):
        text = "Sorry, I didn't quite catch that. I'm happy to explain."
        patterns = detect_hesitation_patterns(text)
        types = [p.pattern_type for p in patterns]
        assert "repair" in types

    def test_empty_text(self):
        patterns = detect_hesitation_patterns("")
        assert patterns == []

    def test_no_hesitation(self):
        text = "I have seven years of experience in full-stack development."
        patterns = detect_hesitation_patterns(text)
        filler_count = sum(1 for p in patterns if p.pattern_type == "filler")
        assert filler_count == 0

    def test_verb_pauses(self):
        text = "Like, you know, I mean, I have worked here."
        patterns = detect_hesitation_patterns(text)
        types = [p.pattern_type for p in patterns]
        assert "pause" in types

    def test_positions_sorted(self):
        text = "Um uh er."
        patterns = detect_hesitation_patterns(text)
        positions = [p.position for p in patterns]
        assert positions == sorted(positions)


# ---- pace measurement ---- #


class TestMeasurePace:
    def test_basic_metrics(self):
        text = "I have seven years of experience in Python and Django."
        pace = measure_pace(text)
        assert pace.word_count >= 10
        assert pace.char_count > 0
        assert pace.pace_label in ("slow", "normal", "fast", "too_slow", "too_fast")
        assert 0.0 <= pace.pace_score <= 1.0
        assert 0.0 <= pace.length_score <= 1.0

    def test_short_answer(self):
        text = "Yes."
        pace = measure_pace(text)
        assert pace.length_label in ("short", "too_short")
        assert pace.length_score < 1.0

    def test_long_answer(self):
        text = " ".join(["Python"] * 150)
        pace = measure_pace(text)
        assert pace.length_label in ("long", "too_long")
        assert pace.length_score < 1.0

    def test_duration_wps(self):
        text = "I have five years of experience in Java."
        pace = measure_pace(text, duration_seconds=3.0)
        assert pace.words_per_second is not None
        assert pace.words_per_second > 0
        # ~5 words / 3 seconds ≈ 1.67 wps
        assert pace.pace_label in ("slow", "normal")

    def test_normal_pace(self):
        text = " ".join(["python"] * 40)
        pace = measure_pace(text, duration_seconds=20.0)
        assert pace.words_per_second == 2.0
        assert pace.pace_label == "normal"
        assert pace.pace_score == 1.0

    def test_empty_text(self):
        pace = measure_pace("")
        assert pace.word_count == 0
        assert pace.length_label == "too_short"

    def test_sentence_split(self):
        text = "I have five years. I know Python well. I led a team."
        pace = measure_pace(text)
        assert pace.sentence_count >= 3


# ---- uncertainty detection ---- #


class TestDetectUncertainty:
    def test_hedge_maybe(self):
        text = "Maybe I have five years, perhaps three."
        signals = detect_uncertainty(text)
        hedge_count = sum(1 for s in signals if s.signal_type == "hedge")
        assert hedge_count >= 2

    def test_doubt_i_dont_know(self):
        text = "I don't know exactly, but I think it was around three years."
        signals = detect_uncertainty(text)
        types = [s.signal_type for s in signals]
        assert "doubt" in types

    def test_vague_quantifiers(self):
        text = "I have worked on several projects, some of them were quite a lot of work."
        signals = detect_uncertainty(text)
        types = [s.signal_type for s in signals]
        assert "vague_quantifier" in types

    def test_i_think(self):
        text = "I think I have around five years."
        signals = detect_uncertainty(text)
        hedge_count = sum(1 for s in signals if s.signal_type == "hedge")
        assert hedge_count >= 1

    def test_confident_answer(self):
        text = "I have seven years of experience. I led a team of twelve engineers."
        signals = detect_uncertainty(text)
        # No hedges or doubts in this confident answer.
        assert len(signals) == 0

    def test_empty_text(self):
        signals = detect_uncertainty("")
        assert signals == []

    def test_positions_sorted(self):
        text = "Maybe perhaps around three years."
        signals = detect_uncertainty(text)
        positions = [s.position for s in signals]
        assert positions == sorted(positions)


# ---- convenience wrapper ---- #


class TestAnalyzeConfidence:
    def test_returns_all_three(self):
        text = "Um, I think I have maybe five years of experience."
        hesitations, pace, uncertainties = analyze_confidence(text)
        assert isinstance(hesitations, list)
        assert isinstance(pace, PaceMetrics)
        assert isinstance(uncertainties, list)
        assert len(hesitations) > 0
        assert len(uncertainties) > 0

    def test_with_duration(self):
        text = "I have five years of Python experience."
        hesitations, pace, uncertainties = analyze_confidence(text, duration_seconds=2.0)
        assert pace.words_per_second is not None
