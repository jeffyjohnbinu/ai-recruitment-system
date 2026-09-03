# Voice Screening Transcript Data Architecture

**Day 23 deliverable — Zecpath AI Job Portal**

Defines how raw voice-interview conversations are converted into
structured, AI-processable data: a per-session voice transcript schema
(raw + normalized text, per-segment metadata) and a derived AI
screening data structure consumable the same way Day 9-12 component
outputs are.

## What it does

1. **Stores** every interview session as a structured object: one
   `VoiceTranscriptRecord` per (candidate, job, session), containing an
   ordered list of `QuestionTranscript`s, each holding an ordered list
   of `TranscriptSegment`s (one per speaker turn / ASR chunk).
2. **Standardizes metadata** on every segment: `candidate_id` /
   `job_id` (record level), `question_id`, `segment_id`, `timestamp_start`
   / `timestamp_end`, and ASR `confidence` — plus `asr_engine` and
   `language` for auditability.
3. **Normalizes** raw ASR text via `TranscriptNormalizer`:
   - strips filler disfluencies (um, uh, "you know") without touching
     real words ("I would *like* to..." stays intact)
   - inserts sentence breaks on long inter-word pause gaps, when
     word-level timestamps are supplied
   - collapses stutters / immediate word repeats ("I I think" -> "I think")
   - capitalizes sentence starts and canonicalizes known technical terms
     via an optional pluggable alias lookup (e.g. Day 9's skill dictionary)
   - **never deletes** low-confidence content — segments below
     `LOW_CONFIDENCE_THRESHOLD` (0.6) are flagged (`low_confidence=True`),
     not dropped, so downstream stages decide what to do with them
   - does **not** scrub PII — that responsibility stays with
     `fairness_bias_engine` (additive-only; no duplicated masking logic)
4. **Derives** an `AIScreeningRecord` per session: per-question
   relevance score, mean ASR confidence, matched keywords, and quality
   flags, plus an overall communication score — shaped to plug into
   `ats_scoring_engine` as an additional weighted component the same
   way Day 9-12 outputs do.
5. **Stores** results as structured JSON, following the same
   `outputs/<subdir>/<name>.json` convention used since Day 5.

## Project layout

```
voice_screening_engine/
├── __init__.py
├── schema.py           # TranscriptSegment / QuestionTranscript / VoiceTranscriptRecord
│                          # + QuestionScreeningResult / AIScreeningRecord
├── normalizer.py         # TranscriptNormalizer -- the 6 normalization rules
├── storage.py               # TranscriptResultStore (JSON output for both record types)
└── tests/
    ├── test_voice_screening.py  # pytest suite (26 tests)
    ├── run_tests.py               # runs the suite + writes a timestamped log
    └── logs/                        # generated test-run logs (deliverable)
```

## Requirements

No third-party runtime dependencies — pure Python (`dataclasses`, `re`,
`json`, `uuid`, `datetime`). Only `pytest` is needed for the test suite:

```
pip install pytest --break-system-packages
```

## Usage

### Python API

```python
from voice_screening_engine import (
    TranscriptSegment, QuestionTranscript, VoiceTranscriptRecord,
    QuestionScreeningResult, AIScreeningRecord,
    TranscriptNormalizer, TranscriptResultStore,
)

store = TranscriptResultStore(output_dir="outputs")
normalizer = TranscriptNormalizer(term_lookup={"python": "Python", "aws": "AWS"})

text, notes = normalizer.normalize("um so like i i worked on python and aws")
# text  -> "So like I worked on Python and AWS."
# notes -> ["stripped 1 filler token(s)", "collapsed 1 repeated word(s)"]

segment = TranscriptSegment(
    segment_id="q1_s1", question_id="q1", speaker="candidate",
    text_raw="um so like i i worked on python and aws",
    text_normalized=text,
    timestamp_start="2026-09-03T10:00:00+00:00",
    timestamp_end="2026-09-03T10:00:05+00:00",
    confidence=0.91, asr_engine="whisper-large-v3",
)

question = QuestionTranscript(
    question_id="q1", question_text="Tell me about your last project.",
    asked_at="2026-09-03T10:00:00+00:00", segments=[segment],
)

transcript = VoiceTranscriptRecord(
    candidate_id="c123", job_id="job_456", session_id="sess_abc",
    generated_at=store.now_iso(), request_id=store.new_request_id(),
    questions=[question],
)

screening = AIScreeningRecord(
    candidate_id="c123", job_id="job_456", session_id="sess_abc",
    generated_at=store.now_iso(), request_id=store.new_request_id(),
    per_question=[QuestionScreeningResult(
        question_id="q1", answer_text=question.candidate_text(),
        relevance_score=0.8, confidence_avg=question.average_confidence(),
        keywords_matched=["Python", "AWS"],
    )],
    overall_communication_score=0.75,
)

paths = store.save_pair(transcript, screening)
print(paths)
# {"transcript": "outputs/transcripts/c123__job_456__sess_abc.json",
#  "screening":  "outputs/structured/c123__job_456__sess_abc.screening.json"}
```

### Output structure

For each interview session, two files are written:
- `outputs/transcripts/<candidate_id>__<job_id>__<session_id>.json` —
  full `VoiceTranscriptRecord` (every question, every segment, raw +
  normalized text, timestamps, confidence)
- `outputs/structured/<candidate_id>__<job_id>__<session_id>.screening.json` —
  full `AIScreeningRecord` (per-question relevance/confidence/flags,
  overall communication score)

Both carry the Day 7 metadata envelope
(`schema_version`, `model_version`, `pipeline_version`, `candidate_id`,
`job_id`, `generated_at`, `request_id`).

## Metadata standard

| Field | Level | Notes |
|---|---|---|
| `candidate_id` | record | Day 7 envelope |
| `job_id` | record | Day 7 envelope |
| `session_id` | record | one interview session |
| `question_id` | segment/question | ties segments back to the asked question |
| `segment_id` | segment | e.g. `"q1_s1"` |
| `timestamp_start` / `timestamp_end` | segment | ISO 8601 |
| `confidence` | segment | ASR confidence in [0, 1], **not** answer quality |
| `asr_engine` | segment | e.g. `"whisper-large-v3"`, `"fallback-engine"` |
| `language` | segment | default `"en"` |
| `low_confidence` | segment | auto-derived flag, `confidence < 0.6` |

## Running tests

```
python voice_screening_engine/tests/run_tests.py
```

This runs the full pytest suite and writes a timestamped log to
`tests/logs/test_run_<UTC timestamp>.log`. 26 tests cover: filler-word
and filler-phrase stripping (including the "real word 'like'" negative
case), pause-based sentence-break insertion (with and without
word-level timestamps), sentence capitalization and technical-term
canonicalization (with and without a supplied lookup), stutter/repeat
collapsing, empty-input handling, low-confidence segment flagging at
and below threshold (confirming text is never dropped), `QuestionTranscript`
helper methods (`candidate_text()`, `average_confidence()`), metadata
envelope presence, and full JSON persistence for both record types.

## Known limitations

- **Filler/term lists are static.** `_FILLER_TOKENS` /
  `_FILLER_PHRASES` and the optional `term_lookup` are plain Python
  data structures, matching the project's "extend the list, don't
  change the logic" convention (same tradeoff as `jd_parsing_engine`'s
  skill/role synonym tables). Unusual disfluencies or unlisted jargon
  pass through untouched rather than being guessed at.
- **Pause-based sentence breaks require word-level timestamps.**
  Without them (a plain ASR transcript string), rule 2 is a no-op —
  the segment's existing punctuation is trusted as-is.
- **No PII masking.** By design this module does not scrub or redact
  personal information from transcript text; that responsibility stays
  with `fairness_bias_engine` downstream so masking logic exists in
  exactly one place in the pipeline.
- **Relevance scoring is a placeholder shape.** `QuestionScreeningResult.relevance_score`
  is a plain field here — this package defines the *data structure* an
  AI screening scorer would populate; the scoring model/prompt itself
  is out of scope for Day 23 and would be wired in as its own
  screening-stage module, consistent with the additive-only
  integration convention used throughout the pipeline.
