# Section Detection Accuracy Report

Day 8 deliverable — Zecpath AI Job Portal

Ground truth: hand-labeled substrings across 5 fixture resumes (full headings, no headings, partial headings, table-linearized skills, noisy/OCR-style spacing).

| Fixture | Correct | Total | Accuracy |
|---|---|---|---|
| resume_full_headings.txt | 7 | 7 | 100% |
| resume_no_headings.txt | 6 | 7 | 86% |
| resume_partial_headings.txt | 4 | 4 | 100% |
| resume_table_skills.txt | 6 | 6 | 100% |
| resume_noisy_ocr.txt | 5 | 5 | 100% |
| **Overall** | **28** | **29** | **97%** |

### Mismatches in `resume_no_headings.txt`
- `Reduced reporting errors by 25%...` expected **Experience**, got **Contact**
