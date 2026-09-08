"""
storage.py
----------
Persists screening reports to disk.

Layout:
    outputs/
      structured/
        screening_reports/
          <candidate_id>__<job_id>__<session_id>.report.json
          <candidate_id>__<job_id>__<session_id>.report.html
          <candidate_id>__<job_id>__<session_id>.report.txt
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .report_format import ScreeningReport


class ScreeningReportStore:
    """
    Writes screening reports to disk.

    Layout:
        outputs/structured/screening_reports/
          <candidate_id>__<job_id>__<session_id>.report.json
          <candidate_id>__<job_id>__<session_id>.report.html
          <candidate_id>__<job_id>__<session_id>.report.txt
    """

    def __init__(self, output_dir: str | Path = "outputs") -> None:
        self.output_dir = Path(output_dir)
        self.report_dir = self.output_dir / "structured" / "screening_reports"
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def save_report(
        self, report: ScreeningReport, formats: list[str] | None = None
    ) -> Dict[str, str]:
        """
        Save a report in multiple formats.

        Args:
            report: The ScreeningReport to persist.
            formats: List of formats to save. Defaults to ['json'].

        Returns:
            Dict mapping format name to file path.
        """
        if formats is None:
            formats = ["json"]

        paths: Dict[str, str] = {}
        base_name = f"{report.candidate_id}__{report.job_id}__{report.session_id}"

        for fmt in formats:
            ext = (
                "report.json"
                if fmt == "json"
                else (
                    "report.html"
                    if fmt == "html"
                    else "report.txt" if fmt == "printable" else "report.email.txt"
                )
            )
            filename = f"{base_name}.{ext}"
            path = self.report_dir / filename

            if fmt == "json":
                path.write_text(
                    json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            elif fmt == "html":
                from .export_formats import HTMLExporter

                html = HTMLExporter().export(report)
                path.write_text(html, encoding="utf-8")
            elif fmt == "printable":
                from .export_formats import PrintableExporter

                txt = PrintableExporter().export(report)
                path.write_text(txt, encoding="utf-8")
            elif fmt == "email":
                from .export_formats import EmailExporter

                email = EmailExporter().export(report)
                path.write_text(email, encoding="utf-8")

            paths[fmt] = str(path)

        return paths

    def save_json(self, report: ScreeningReport) -> str:
        """Save report as JSON only."""
        paths = self.save_report(report, formats=["json"])
        return paths["json"]

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def new_request_id() -> str:
        import uuid

        return str(uuid.uuid4())
