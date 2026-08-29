# Day 16 — ATS API Design & Integration Planning

**Zecpath AI Job Portal — planning deliverable (no runtime code this day).**

Designs the REST surface over the Day 5–14 pipeline: resume upload, JD
submission, matching, and shortlisting — with async job handling, error
standards, and logging standards. Nothing here is wired to the existing
modules yet; that integration work is scoped as a future day.

## Contents

```
ats_api_design/
├── openapi.yaml                          # OpenAPI 3.0.3 spec (validated)
├── Day16_ATS_API_Specification.docx      # Design rationale doc
├── Day16_ATS_API_Specification.pdf       # Rendered copy, for quick review
├── schemas/                              # Standalone JSON Schema (draft-07) per contract
│   ├── resume_record.schema.json
│   ├── job_requirement_record.schema.json
│   ├── match_create_request.schema.json
│   ├── match_result.schema.json
│   ├── shortlist_create_request.schema.json
│   ├── shortlist_result.schema.json
│   ├── task.schema.json
│   └── error_response.schema.json
└── examples/                             # Example payloads, validated against the schemas above
    ├── task_processing.json
    ├── match_result.json
    └── error_response.json
```

## Validating the spec yourself

```bash
pip install openapi-spec-validator jsonschema --break-system-packages
python -m openapi_spec_validator openapi.yaml
python -c "import json,jsonschema; jsonschema.validate(json.load(open('examples/match_result.json')), json.load(open('schemas/match_result.schema.json')))"
```

Or preview interactively with Swagger UI / Redoc:

```bash
npx @redocly/cli preview-docs openapi.yaml
```

## Key design decisions

- **One Task model for every async op** (resume upload, JD parsing when large,
  matching, shortlisting) rather than a bespoke status shape per endpoint —
  polling and webhook handling are identical everywhere.
- **Contracts mirror existing dataclasses** — `ResumeRecord` maps onto
  `ExtractionRecord` (Day 5) plus the Day 8–11 enrichment fields;
  `MatchResult` mirrors the Day 13 ATS Scoring Engine output, including the
  `weakest_component` field computed from raw score (the Day 13 explainability
  fix); `ShortlistResult` mirrors the Day 14 zone/rank/tie-break model.
- **Errors are RFC 7807 problem+json**, extended with a `code` enum and a
  `trace_id` that also appears in server logs for correlation.
- **Idempotency-Key** on every resource-creating POST, since resume uploads
  and batch shortlist requests are exactly the kind of calls integrators
  retry on timeout.

Full rationale for async job handling, error codes, and the logging standard
is in `Day16_ATS_API_Specification.docx`.
