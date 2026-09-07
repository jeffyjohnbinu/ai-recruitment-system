# Day 27 — Confidence & Sentiment Signal Analysis

**Zecpath AI Job Portal — Phase 16/17, Phase 40, Phase 65**

Measures communication quality and behavioral indicators from interview answers.

---

## What It Does

Scores how candidates communicate during AI screening calls across five signals:

| Signal | What It Detects |
|--------|----------------|
| **Hesitation patterns** | Filler words (`um`, `uh`), repetitions, stutters, repairs (`I mean`, `or rather`), false starts |
| **Pace metrics** | Words-per-second, sentence length, pace labels (`normal` / `slow` / `fast` / `too_slow` / `too_fast`) |
| **Uncertainty signals** | Hedging (`maybe`, `I think`), doubt phrases (`I don't know`), vague quantifiers (`several`, `around`) |
| **Sentiment polarity** | Positive / negative / neutral / mixed tone; emotion signals (enthusiasm, frustration, confidence, anxiety, calm, pride) |
| **Self-conversations** | Internal contradictions (close-proximity opposing statements); cross-answer conflicts (availability, salary) |

Combines all signals into a **Communication Strength Indicator** (clarity, confidence, conviction, engagement, professionalism) and aggregates to a session-level **Behavioral Indicators Report**.

---

## Files

```
day27_confidence_sentiment/
├── __init__.py              # Public API exports
├── formats.py               # Output dataclasses (HesitationPattern, PaceMetrics,
│                            #   UncertaintySignal, SentimentScore,
│                            #   CommunicationStrengthIndicator, ConfidenceAnalysis,
│                            #   BehavioralIndicatorsReport)
├── confidence_analyzer.py    # Hesitation, pace, uncertainty detection (pure functions)
├── sentiment_scorer.py      # Polarity, emotion, contradiction detection (pure functions)
├── engine.py                # ConfidenceSentimentEngine orchestrator
├── cli.py                   # CLI entry point
└── tests/
    ├── test_confidence_analyzer.py
    ├── test_sentiment_scorer.py
    └── test_engine.py
```

---

## Quick Start

```bash
# Demo (6 sample answers)
python -m day27_confidence_sentiment.cli --demo

# JSON output
python -m day27_confidence_sentiment.cli --demo --json-output

# Custom answers
python -m day27_confidence_sentiment.cli \
    --candidate cand_001 \
    --job job_backend_01 \
    --session sess_001 \
    --role software_engineer \
    --answers q1,experience,"I have 5 years of Python." \
    --answers q2,salary,"My expected is 15 LPA."
```

---

## Python API

```python
from day27_confidence_sentiment import ConfidenceSentimentEngine

engine = ConfidenceSentimentEngine()

# One answer
result = engine.score_answer("q1", {
    "raw_answer": "I have five years of experience in Python and Django.",
    "category": "experience",
    "duration_seconds": 4.0,   # optional
})

# Full session
session = engine.score_session(
    candidate_id="cand_001",
    job_id="job_backend_01",
    session_id="sess_001",
    role_id="software_engineer",
    answers=[
        ("q1", {"raw_answer": "I have five years of experience.", "category": "experience"}),
        ("q2", {"raw_answer": "My expected salary is 18 LPA.", "category": "salary"}),
    ],
)

# Build full report
report = engine.build_behavioral_report(session)
print(report.to_dict())
```

---

## Key Design Decisions

### Pure rule-based
All scoring is done with regex lexicons and heuristics. No LLM call required. LLM stubs (`_llm_score_sentiment`, `_llm_detect_contradictions`) are provided for future upgrade — they currently fall back to rules.

### Day 7 metadata envelope
Every output record carries `schema_version`, `model_version` (`day27-v1`), `pipeline_version`, `generated_at`, and `request_id`.

### Communication strength dimensions

| Dimension | Primary Signal | Weight |
|-----------|---------------|--------|
| `clarity` | Hesitation density + pace score | 25% |
| `confidence` | Uncertainty density + hesitation severity | 25% |
| `conviction` | Polarity decisiveness | 20% |
| `engagement` | Answer length + pace | 15% |
| `professionalism` | Absence of heavy hedges + sentiment | 15% |

### Strength labels
- **exceptional** ≥ 90%
- **strong** ≥ 75%
- **competent** ≥ 60%
- **developing** ≥ 40%
- **weak** < 40%

### Contradiction detection
- **Internal**: Regex patterns for close-proximity contradictory pairs (within 80 chars).
- **Cross-answer**: Lightweight keyword overlap for availability conflicts. Full semantic cross-answer detection is in the `_llm_detect_contradictions` stub.

---

## Integration Points

| Source | Day Module | Data Used |
|--------|-----------|-----------|
| STT / Transcript | Day 24 (`day24_stt`) | `raw_answer` text |
| Answer Understanding | Day 25 (`answer_understanding_engine`) | `category`, optional `duration_seconds` |
| Screening Scoring | Day 26 (`day26_screening_scoring_engine`) | Output enriches `SessionScore` with behavioral signals |

The `BehavioralIndicatorsReport` can be embedded in the Day 26 `SessionScore` as an additional dimension for a comprehensive candidate profile.

---

## Thresholds

| Threshold | Value |
|-----------|-------|
| High hesitation | > 4 patterns / answer |
| High uncertainty | > 3 signals / answer |
| Low confidence | < 50% overall score |
| Negative sentiment | polarity < −0.3 |
| Pace — too slow | < 1.5 wps |
| Pace — too fast | > 3.0 wps |
| Ideal answer length | 3–60 words |

---

## Version Info

- `schema_version`: 1.0.0
- `model_version`: day27-v1
- `pipeline_version`: 1.0.0
