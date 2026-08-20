# Skill Extraction Engine

**Day 9 deliverable — Zecpath AI Job Portal**

Extracts technical, business, and creative skills from resume text,
resolving synonyms, spelling variations, and multi-skill "stacks" (MERN,
MEAN, LAMP...) into a deduplicated, confidence-scored, structured output
that downstream stages (ATS matching, AI screening) can consume directly.

## What it does

1. **Matches** resume text against a master skill dictionary spanning
   three categories — `technical`, `business`, `creative` — using three
   passes:
   - **Exact / alias phrase matching**: canonical names and known
     synonyms/abbreviations ("ReactJS" → `React`, "Node JS" → `Node.js`),
     longest-phrase-first so multi-word skills win over shorter overlaps.
   - **Skill-stack expansion**: a single mention of a stack token (e.g.
     "MERN stack") expands into its constituent skills (`MongoDB`,
     `Express.js`, `React`, `Node.js`), tagged `stack_inferred` so they
     can be scored lower than a directly-stated skill.
   - **Fuzzy spelling-variant matching**: any word not already matched is
     compared against the dictionary with `rapidfuzz` to catch typos and
     glyph substitutions ("Kubernets" → `Kubernetes`, "Pyth0n" → `Python`),
     with a curated stopword list and length/ratio thresholds to keep
     ordinary English words from false-matching short skill names.
2. **Scores confidence** per skill (0–1), explainably: a base score by
   match type (exact > alias > stack-inferred, fuzzy scaled from its
   similarity ratio), plus a small bonus if the skill was mentioned inside
   a skills-bearing section (Skills / Certifications / Projects) and a
   small bonus for repeated mentions.
3. **Deduplicates and normalizes** every raw mention into one record per
   canonical skill — merging match types (the strongest one wins),
   mention counts, matched text variants, and source sections.
4. **Stores** results as structured JSON, in the same style as Day 5's
   `resume_extraction_engine` and Day 6's `jd_parsing_engine`.

## Project layout

```
skill_extraction_engine/
├── __init__.py
├── skill_dictionary.py    # master dictionary: technical/business/creative + aliases + stacks
├── matcher.py               # SkillMatcher: phrase / stack / fuzzy matching passes
├── confidence.py             # per-skill confidence scoring
├── normalizer.py              # dedup raw mentions -> SkillRecord list
├── storage.py                  # SkillExtractionResult + SkillResultStore (JSON output)
├── extractor.py                  # main orchestrator (SkillExtractionEngine)
├── cli.py                          # command-line entry point
└── tests/
    ├── test_skill_extraction.py     # pytest suite (23 tests)
    ├── run_tests.py                   # runs the suite + writes a timestamped log
    ├── fixtures/                       # 2 sample cleaned-resume text fixtures
    └── logs/                             # generated test-run logs (deliverable)
```

## Requirements

```
pip install rapidfuzz pytest --break-system-packages
```

## Usage

### Python API

```python
from skill_extraction_engine import SkillExtractionEngine

engine = SkillExtractionEngine(output_dir="outputs")

# Works directly on Day 5's `cleaned_text` output
result = engine.extract_from_text(cleaned_text, source_name="john_doe.docx")

print(result.status)              # "success" | "partial" | "failed"
print(result.total_skills_found)  # e.g. 12
for skill in result.skills:
    print(skill.skill, skill.category, skill.confidence, skill.match_type)
```

Or read a saved cleaned-text file directly:
```python
result = engine.extract_from_file("cleaned_text/john_doe.clean.txt")
```

### CLI

```
python -m skill_extraction_engine.cli cleaned_text/john_doe.clean.txt
python -m skill_extraction_engine.cli cleaned_text/ --output-dir extracted_skills/
```

### Output structure

For each resume, one file is written:
- `outputs/structured/<name>.skills.json` — full `SkillExtractionResult`
  (per-skill category, confidence, match type, mention count, matched
  text variants, and source sections)

Example record:
```json
{
  "skill": "Kubernetes",
  "category": "technical",
  "subcategory": "devops",
  "confidence": 0.8526,
  "match_type": "fuzzy",
  "mention_count": 1,
  "matched_variants": ["Kubernets"],
  "source_sections": ["Experience"],
  "stack_sources": []
}
```

## Running tests

```
python skill_extraction_engine/tests/run_tests.py
```

23 tests cover: exact/alias/multi-word matching across all three
categories, skill-stack expansion and its lower confidence tier, fuzzy
spelling-variant matching (including a regression guard against
common-English-word false positives), deduplication/merge-by-match-type
behavior, confidence bounds and the section/frequency bonuses, empty-input
and no-match status handling, structured JSON output, and dictionary
integrity (no duplicate aliases, full category coverage).

## Pipeline integration

- **Upstream**: consumes Day 5's `cleaned_text` (and optionally Day 8's
  detected section labels for context) — no changes required to either
  module.
- **Downstream**: `SkillExtractionResult.skills` is a flat, JSON-serializable
  list ready for ATS keyword/embedding matching or AI screening prompts.

## Known limitations

- **Dictionary coverage**: the master dictionary (~110 skills across the
  three categories) covers common software/business/creative roles well,
  but is not exhaustive — niche or emerging tools need a dictionary entry
  added before they can be recognized. Extending it is a one-line addition
  to `skill_dictionary.py`.
- **Fuzzy matching is conservative by design**: the similarity threshold
  and stopword list favor precision over recall, so unusual misspellings
  beyond a couple of edited characters may go unmatched rather than risk
  false positives.
- **Section detection** relies on Day 5's canonical heading normalization
  (`Skills`, `Experience`, ...) being present as standalone lines in the
  cleaned text; resumes without clear section headers still get correctly
  matched skills, just without the section confidence bonus.
