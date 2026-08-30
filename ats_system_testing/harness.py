"""
harness.py
-----------
Runs the Day 17 fixture set through the ATS pipeline (via
pipeline_adapter) and packages each result alongside its ground truth,
ready for metrics.py and storage.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .fixtures import ATSTestCase, get_test_cases
from .pipeline_adapter import PipelineOutput, run_ats_pipeline


@dataclass
class CaseResult:
    test_case: ATSTestCase
    pipeline_output: PipelineOutput

    @property
    def is_correct(self) -> bool:
        return self.pipeline_output.shortlisted == self.test_case.ground_truth.shortlisted


class ATSTestHarness:
    """Orchestrates running fixtures through the pipeline adapter."""

    def __init__(self, test_cases: Optional[List[ATSTestCase]] = None):
        self.test_cases = test_cases if test_cases is not None else get_test_cases()

    def run(self) -> List[CaseResult]:
        results = []
        for case in self.test_cases:
            output = run_ats_pipeline(case.resume_text, case.job_description)
            results.append(CaseResult(test_case=case, pipeline_output=output))
        return results
