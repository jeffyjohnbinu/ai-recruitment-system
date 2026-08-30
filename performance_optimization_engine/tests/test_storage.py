import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from performance_optimization_engine.storage import (  # noqa: E402
    PerformanceReport,
    ReportStore,
    build_metadata_envelope,
)


def test_metadata_envelope_has_required_fields():
    envelope = build_metadata_envelope()
    for field in (
        "schema_version",
        "model_version",
        "pipeline_version",
        "candidate_id",
        "job_id",
        "generated_at",
        "request_id",
    ):
        assert field in envelope


def test_metadata_envelope_request_id_unique():
    e1 = build_metadata_envelope()
    e2 = build_metadata_envelope()
    assert e1["request_id"] != e2["request_id"]


def test_report_store_writes_json(tmp_path):
    store = ReportStore(tmp_path)
    report = PerformanceReport(metadata=build_metadata_envelope(), benchmark={"ok": True})
    path = store.save(report)

    assert Path(path).exists()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["benchmark"]["ok"] is True
    assert "metadata" in data


def test_report_store_creates_output_dir(tmp_path):
    target = tmp_path / "nested" / "dir"
    ReportStore(target)
    assert target.exists()
