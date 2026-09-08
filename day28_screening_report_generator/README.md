# Day 28: AI Screening Report Generator

## Overview

**AI Screening Report Generator** transforms raw AI evaluations from **Day 26 (Scoring Engine)** and **Day 27 (Behavioral Analysis)** into comprehensive, recruiter-friendly reports.

This deliverable completes the end-to-end automation pipeline by converting complex AI scoring and behavioral data into actionable insights for recruiters.

## Core Features

### 1. Structured Report Generation
- Combines scoring results (Day 26 `SessionScore`) with behavioral insights (Day 27 `BehavioralIndicatorsReport`)
- Extracts key answers, strengths, risks, and missing data
- Highlights salary expectations, availability status, and skill confirmations

### 2. Multiple Export Formats
- **JSON**: Machine-readable structured format
- **HTML**: Browser-ready interactive reports
- **Printable**: Terminal/print-friendly text format
- **Email**: Email-friendly summary format

### 3. Recruiter-Facing Insights
- **Key Answers**: Extracted and summarized candidate responses
- **Strength Profiles**: Aggregated communication and technical strengths
- **Risk Profiles**: Identified behavioral and communication risks
- **Compensation Insights**: Extracted salary and availability data
- **Skill Confirmation**: Matched candidate skills against job requirements

## Architecture

```
Day 26 Scoring Engine  →  SessionScore  →  ScreeningReportBuilder  →  ScreeningReport
                               ↑
Day 27 Behavioral Analysis  →  BehavioralIndicatorsReport
                                             ↓
                                         Multiple Exporters  →  Various Formats
```

### Key Components

1. **`report_format.py`** - Dataclasses for report structure
   - `ScreeningReport` - Complete report object
   - `KeyAnswer` - Individual Q&A records
   - `StrengthProfile` - Candidate strengths summary
   - `RiskProfile` - Risk and concern profile
   - `CompensationInsights` - Salary and availability data
   - `SkillConfirmation` - Extracted skills and experience

2. **`report_builder.py`** - Core report builder
   - Combines Day 26 and Day 27 outputs
   - Extracts key insights using pattern matching
   - Builds recruiter narratives and summaries
   - Handles data validation and cleaning

3. **`export_formats.py`** - Multiple export options
   - `JSONExporter` - Structured JSON export
   - `HTMLExporter` - Web-ready HTML reports
   - `PrintableExporter` - Terminal/print-friendly text
   - `EmailExporter` - Email summary format

4. **`storage.py`** - Persistent storage
   - Saves reports in multiple formats
   - Organized by candidate/job/session
   - Atomic writes for reliability

5. **`cli.py`** - Command-line interface
   - Process Day 26 and Day 27 JSON files
   - Generate reports in selected formats
   - Built-in demo mode for testing

## Usage

### Basic Command (Generate Reports)

```bash
# Generate all formats
python -m day28_screening_report_generator.cli \
    --score outputs/structured/screening_scores/cand_001__job_01__sess_001.score.json \
    --behavioral outputs/structured/behavioral_reports/cand_001__job_01__sess_001.behavioral.json \
    --format json html printable email \
    --output-dir outputs/structured/screening_reports
```

### Demo Mode

```bash
# Run with built-in sample data
python -m day28_screening_report_generator.cli --demo
```

### Programmatic Usage

```python
from day28_screening_report_generator import ScreeningReportBuilder

builder = ScreeningReportBuilder()
report = builder.build_report(
    session_score=session_score_obj,
    behavioral_report=behavioral_report_obj,
    request_id="req_001",
)

# Export in different formats
from day28_screening_report_generator import get_exporter

json_exporter = get_exporter("json")
json_str = json_exporter.export(report)

html_exporter = get_exporter("html")
html_str = html_exporter.export(report)

# Recruiter summary
print(report.to_recruiter_summary())
```

## Integration with Day 26 and Day 27

### Day 26 Integration
The builder expects Day 26 `SessionScore` objects (or their dict representations) with these required fields:
- `candidate_id`, `job_id`, `session_id`, `role_id`
- `normalized_score`, `recommendation`
- `breakdown` - List of per-question scoring results
- `overall_dimension_scores` - Four-dimensional scoring
- `consistency` - Cross-question consistency check

### Day 27 Integration
The builder expects Day 27 `BehavioralIndicatorsReport` objects (or their dict representations) with these required fields:
- `candidate_id`, `job_id`, `session_id`, `role_id`
- `session` - Session-level aggregates
- `per_answer` - Per-answer analysis
- `strength_distribution`, `session_avg_strength`, `session_avg_confidence`

## Report Components Explained

### Key Answers
Extracted from the most relevant questions for recruiter insight:
- Salary expectations and confidence
- Notice periods and availability
- Technical experience and skills
- Communication strengths and concerns

### Strength Profile
- **Communication Strength**: Overall communication effectiveness
- **Clarity**: Clear and articulate responses
- **Confidence**: Assertiveness and conviction
- **Conviction**: Decision-making and certainty
- **Engagement**: Interest and participation level
- **Professionalism**: Professional demeanor and tone

### Risk Profile
- **Hesitation Level**: Nervousness or uncertainty in responses
- **Uncertainty Level**: Vague or non-committal answers
- **Sentiment Risk**: Negative emotional tone
- **Contradictions**: Inconsistent statements
- **Red Flag Count**: Number of flagged answers

### Compensation Insights
- Extracted from salary-related questions
- Confidence levels for salary data
- Notice periods and availability status
- Location preferences and relocation willingness

### Skill Confirmation
- Skills explicitly mentioned in responses
- Experience level assessment
- Confidence scores for each skill
- Identified skill gaps

## Export Formats

### JSON Format
```json
{
  "schema_version": "1.0.0",
  "model_version": "screening-report-generator-1.0.0",
  "candidate_id": "cand_001",
  "job_id": "job_01",
  "overall_score": 0.85,
  "recommendation": "proceed",
  "strength_profile": {"communication_strength_score": 0.88},
  "key_answers": [...],
  "recruiter_narrative": "Strong candidate..."
}
```

### HTML Format
Browser-ready interactive report with:
- Responsive design for desktop/tablet/mobile
- Color-coded scores and risk indicators
- Professional styling with modern CSS
- Printable-friendly layout

### Printable Format
Terminal and print-friendly text format:
- Clear section headers
- Concise bullet points
- Easy-to-read hierarchical structure
- No external dependencies

### Email Format
Summary optimized for email:
- Subject line with key information
- Concise narrative
- Highlighted key metrics
- Actionable recommendations

## Testing

Run the test suite with pytest:

```bash
# Run all tests
pytest day28_screening_report_generator/tests/

# Run specific test modules
pytest day28_screening_report_generator/tests/test_report_format.py
pytest day28_screening_report_generator/tests/test_report_builder.py
pytest day28_screening_report_generator/tests/test_export_formats.py
pytest day28_screening_report_generator/tests/test_storage.py
```

## Files and Structure

```
day28_screening_report_generator/
├── __init__.py                    # Package initialization
├── report_format.py               # Report data structures
├── report_builder.py              # Core report builder
├── export_formats.py              # Exporters for different formats
├── storage.py                     # Persistent storage
├── cli.py                         # Command-line interface
├── sample_reports.py              # Built-in demo data
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_report_format.py
│   ├── test_report_builder.py
│   ├── test_export_formats.py
│   └── test_storage.py
└── README.md                      # This file
```

## Sample Outputs

See `sample_reports.py` for built-in demo data and `tests/` directory for comprehensive test coverage.

## Dependencies

Required Python packages:
- Python 3.7+
- Standard library modules only

No external dependencies - the module is designed to work with any Day 26/Day 27 implementation that follows the expected data structures.

## Future Enhancements

1. **Template Engine** - Customizable report templates
2. **Email Integration** - Direct email delivery
3. **API Endpoint** - REST API for report generation
4. **Dashboard Integration** - Real-time report viewing
5. **Version Control** - Report history and comparison

## Conclusion

The **Day 28 AI Screening Report Generator** successfully transforms complex AI scoring and behavioral analysis data into recruiter-friendly insights. By providing multiple export formats and comprehensive reporting, it enables recruiters to quickly assess candidates and make informed hiring decisions without needing to understand the underlying AI technical details.