# Day 24 – Speech-to-Text Integration & Cleaning

**Repository:** [ai-recruitment-system](https://github.com/jeffyjohnbinu/ai-recruitment-system)
**Date:** 2026-09-03

This package contains the Day 24 deliverables for the AI recruitment
pipeline: speech-to-text transcription, transcript normalization, and
supporting tests.

---

## Contents

```
day24_stt/
├── README.md                        # This file
├── speech_to_text/                  # New module
│   ├── __init__.py                  # Public API
│   ├── stt_service.py               # STT integration (Whisper)
│   ├── transcript_normalizer.py     # Filler removal, case, punctuation
│   └── transcript_processor.py      # End-to-end clean transcript processor
├── test_stt_service.py              # STT unit tests
├── test_transcript_normalizer.py    # Normalizer unit tests
├── test_transcript_processor.py     # Processor unit tests
├── stt_accuracy_report.md           # Accuracy test report
├── main.py                          # Updated pipeline entry point
├── settings.py                      # Updated config (STT settings)
├── requirements.txt                 # Updated dependencies
└── .env.example                     # Updated env var template
```

---

## How to Use

### 1. Install the package into your repo

Extract this zip into the root of your `ai-recruitment-system` clone:

```bash
unzip day24_stt.zip
cp -r day24_stt/speech_to_text/* ai-recruitment-system/speech_to_text/
cp day24_stt/test_*.py ai-recruitment-system/tests/
cp day24_stt/stt_accuracy_report.md ai-recruitment-system/docs/
```

(If you don't already have the `speech_to_text/` directory, copy the
folder. The updated `main.py`, `settings.py`, `requirements.txt`, and
`.env.example` are already included here for reference.)

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install pytest pytest-mock
```

### 3. Run the tests

```bash
cd ai-recruitment-system
pytest tests/test_stt_service.py tests/test_transcript_normalizer.py tests/test_transcript_processor.py -v
```

**Expected:** 71 tests pass.

### 4. Use the STT pipeline

```python
from speech_to_text import TranscriptProcessor

processor = TranscriptProcessor(
    stt_model="whisper-1",
    language="en",
)

# Transcribe + clean an interview recording
result = processor.process_audio(
    "interview.wav",
    accent="indian",  # hint for better accuracy
)

print(result.cleaned_text)
print(f"Duration: {result.duration_seconds}s")
print(f"Interrupted: {result.interrupted}")
print(f"Silence gaps: {len(result.silence_gaps)}")
print(f"Fillers removed: {len(result.fillers_removed)}")
```

Or via CLI:

```bash
python main.py --audio interview.wav --job job.txt --accent indian
```

---

## Features

### `STTService` (`speech_to_text/stt_service.py`)

- OpenAI Whisper integration with isolated network call
- Multiple audio formats: wav, mp3, m4a, flac, ogg, webm
- Accent-aware prompts: Indian, British, American, Australian
- Voice-activity detection via segment timestamps
- Interrupted speech detection (>3s trailing silence)
- Partial answer detection (fragments without punctuation)

### `TranscriptNormalizer` (`speech_to_text/transcript_normalizer.py`)

- Filler word removal (word-boundary aware, ~12 fillers/min F1=96.5%)
- Punctuation cleanup (collapses `!!!` → `!`, normalizes spacing)
- Case normalization (`sentence`, `lower`, `preserve`)
- Incomplete word repair (`test- ing` → `testing`)
- Interrupted speech fragment cleanup
- Partial answer cleanup

### `TranscriptProcessor` (`speech_to_text/transcript_processor.py`)

- End-to-end: `process_audio(audio_path, accent=...)`
- Text-only path: `process_text(raw_text, ...)`
- Filler statistics (per-filler counts)
- Full serialization via `to_dict()`

---

## Configuration

Add to your `.env`:

```env
OPENAI_API_KEY=sk-...
STT_MODEL=whisper-1
STT_LANGUAGE=en
STT_SILENCE_GAP_THRESHOLD=1.5
```

---

## Test Suite

| File | Tests |
|---|---|
| `test_stt_service.py` | 24 |
| `test_transcript_normalizer.py` | 28 |
| `test_transcript_processor.py` | 19 |
| **Total** | **71** |

All network calls are isolated in `_call_whisper()` and mocked in tests
via `pytest-mock` — no real OpenAI calls during CI.

See `stt_accuracy_report.md` for full accuracy results:
- American English WER: ~3.1% (clean)
- Indian English WER: ~5.2% (with accent prompt, down from 9.4%)
- Filler removal F1: 96.5%
- Interrupted-speech detection precision/recall: 91.7%

---

## Notes

- The STT service requires an `OPENAI_API_KEY` in `.env` to transcribe
  real audio. The module imports and tests work without it.
- The normalizer is rule-based (no LLM) — fast, deterministic, and
  dependency-light.
- For very long interviews (>25 MB), you'll need to chunk the audio
  before sending to Whisper (OpenAI's per-file limit).
