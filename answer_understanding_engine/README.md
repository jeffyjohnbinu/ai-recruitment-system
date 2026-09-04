# Answer Intent & Understanding Engine (Day 25)

Turns a candidate's free-form reply to a screening question into a
structured semantic object: intent label, slot values, confidence, and
quality flags (off-topic / vague / missing).

## What it does

Given a raw answer from a screening call, the engine:

1. **Classifies intent** — is this an *answer*, a *clarification
   request*, an *objection*, a *redirect*, *off-topic*, or *no response*?
2. **Extracts slots** — years of experience, notice period, salary,
   skills, city, join-by date, boolean yes/no.
3. **Detects quality issues** — off-topic, vague, missing, low
   completeness.
4. **Returns a `StructuredAnswer`** — a JSON-serialisable semantic object.

## Public API

```python
from answer_understanding_engine.engine import AnswerUnderstandingEngine

engine = AnswerUnderstandingEngine()
answer = engine.understand(
    "I have 5 years of experience in Python and Django.",
    question_id="Q-SE-006",
    expected_slot="years_experience",
)
print(answer.intent.label)              # "answer"
print(answer.extracted["years_experience"])  # 5.0
print(answer.completeness)              # 1.0
```

## Data model

| Object | Fields |
|---|---|
| `IntentResult` | `label`, `confidence`, `rationale` |
| `Slot` | `name`, `value`, `raw_span`, `confidence` |
| `StructuredAnswer` | `question_id`, `raw_answer`, `intent`, `slots`, `extracted`, `is_off_topic`, `is_vague`, `is_missing`, `completeness`, `warnings` |

## Intent labels

- `answer` — on-topic reply
- `clarification_request` — reply is itself a question
- `objection` — candidate raised a concern
- `redirect` — candidate tried to change the subject
- `off_topic` — clearly unrelated reply
- `no_response` — empty / filler / "I don't know"

## Slot vocabulary

| Slot | Example | Example value |
|---|---|---|
| `boolean` | "Yes, I am willing." | `True` |
| `years_experience` | "5 years" | `5.0` |
| `notice_period_days` | "30 days" | `30` |
| `availability` | "I can join immediately." | `"immediate"` |
| `join_by_date` | "15/03/2026" | `"15/03/2026"` |
| `expected_salary_lakhs` | "18 LPA" | `18.0` |
| `skills` | "Python, React, AWS" | `["python", "react", "aws"]` |
| `city` | "I'm based in Bangalore." | `"bengaluru"` |

## LLM path

The engine is deterministic by default. To enable an LLM-based slot
extraction pass, pass `use_llm=True`:

```python
engine = AnswerUnderstandingEngine(use_llm=True)
```

The LLM call is isolated in `_call_llm`, mirroring
`screening_ai.screener._call_llm`, so it can be monkeypatched in tests.

## CLI

```bash
python -m answer_understanding_engine.cli \
    "I have 5 years of experience in Python and Django." \
    --question-id Q-SE-006 --expected-slot years_experience
```

## Tests

```bash
pytest answer_understanding_engine/tests/test_engine.py -v
```

## Integration

This engine is designed to sit between the HR screening question bank
(`hr_screening_question_bank/`) and the scoring layer
(`ats_scoring_engine/`). Each question in the bank has an
`expected_answer_type` and a `category`; the engine maps that category
to an `expected_slot` (see `hr_screening_question_bank/2_question_category_mapping.json`)
and flags answers that fail to deliver the expected slot.