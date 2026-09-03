# HR Screening Question Bank

An AI-ready, structured question bank for automated HR screening calls. Built around 7 fixed categories — Introduction, Education, Experience, Skills, Location, Salary, Notice Period — with every question tagged for answer type, mandatory status, and scoring importance.

## Files in this package

### 1. `1_hr_screening_question_dataset.json`
The core dataset: instantiated, ready-to-ask questions for 4 sample roles (Software Engineer, Sales Executive, Customer Support Executive, Marketing Executive).

Each question object has:
- `question_id` — unique id, prefixed by role (e.g. `Q-SE-001` for Software Engineer)
- `role_id` — which role this question belongs to
- `category` — one of the 7 fixed categories
- `text` — the question text (may contain `{placeholders}` like `{candidate_name}`, `{company_name}`, `{job_location}`, `{work_mode}` to be filled at call time)
- `expected_answer_type` — `text | number | boolean | enum | date | duration`
- `mandatory` — whether the candidate must answer before moving on
- `scoring_weight` — 1 (informational) to 5 (hard filter / auto-reject if unmet)

To add a new role: pick a new prefix (e.g. `Q-DA-` for Data Analyst), write questions following the same shape, and register the role in `2_question_category_mapping.json` where relevant.

### 2. `2_question_category_mapping.json`
Maps each of the 7 categories to:
- its purpose (why it's being asked)
- its position in the call flow (`flow_position`, 1–7)
- the full list of `question_ids` from the dataset that fall under it

Also holds the shared `scoring_scale` (1–5 definitions) and `answer_types` vocabulary, kept in sync with file 1.

### 3. `3_ai_conversation_ready_objects.json`
The schema an AI screening agent uses to actually run a call, turn by turn. Includes:
- `object_schema` — the shape of one conversational turn (resolved question text, validation rule, routing logic for what to ask next based on the answer or a non-response)
- `example_flow_software_engineer` — a worked 5-question example session for one candidate
- `session_summary_object_example` — how answered questions roll up into a score and a recommendation (e.g. `proceed_to_next_round`)

## How the files relate

```
category mapping (2)  --category_id-->  question dataset (1)  --question_id-->  conversation objects (3)
```

`question_id` and `role_id` are the shared keys across all three files, so they join cleanly if loaded into a database, vector store, or used directly as an AI agent's prompt/state source.

## Still to do / extend

- Fill in real values for the `translations` fields (currently stubbed for `en`/`hi`/`es`) once a localization pass is scoped.
- Add more roles to the dataset following the existing `question_id` prefix pattern.
- Wire `routing` rules in file 3 into whatever orchestration layer (IVR, LLM agent, etc.) is running the actual calls.
