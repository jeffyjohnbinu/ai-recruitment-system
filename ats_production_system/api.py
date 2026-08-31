"""
api.py
-------
Day 20 deliverable — Zecpath AI Job Portal

A working FastAPI implementation of the *core* matching flow described in
the Day 16 API design (ats_api_design/openapi.yaml): upload a resume,
upload a job description, request a match, and pull a ranked shortlist.

Scope note (documented honestly rather than silently, per project
convention): this implements the synchronous core of the spec --
/resumes, /job-descriptions, /matches, and
/job-descriptions/{id}/shortlist. It does NOT yet implement every path in
openapi.yaml (PATCH/update endpoints, async task polling via /tasks/{id},
or PDF/CSV shortlist export) -- those are listed as explicit gaps in the
Day 20 evaluation report rather than left for someone to discover later.

Run with:
    uvicorn ats_production_system.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from .pipeline import ATSPipeline

APP_OUTPUT_DIR = Path("outputs")
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Zecpath ATS AI — Production Service",
    version="1.0.0",
    description=(
        "Day 20 reference implementation of the ATS AI microservice: "
        "resume/job ingestion, semantic matching, explainable scoring, "
        "ranking, and fairness auditing."
    ),
)

_pipeline: Optional[ATSPipeline] = None
_resumes: Dict[str, Path] = {}
_jobs: Dict[str, Dict] = {}
_job_files: Dict[str, Path] = {}
_matches: Dict[str, Dict] = {}


def get_pipeline() -> ATSPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ATSPipeline(output_dir=str(APP_OUTPUT_DIR), embedder_preference="auto")
    return _pipeline


class MatchRequest(BaseModel):
    resume_id: str
    job_description_id: str
    role: Optional[str] = None


@app.get("/health")
def health() -> dict:
    pipeline = get_pipeline()
    status = pipeline.engine_status()
    core_ready = all(
        status.get(k)
        for k in ["resume_extraction", "jd_parsing", "ats_scoring", "semantic_matching"]
    )
    return {"status": "ok" if core_ready else "degraded", "engines": status}


@app.post("/resumes", status_code=201)
async def upload_resume(file: UploadFile = File(...)) -> dict:
    ext = Path(file.filename).suffix.lower()
    if ext not in {".pdf", ".docx"}:
        raise HTTPException(400, f"Unsupported resume file type '{ext}'. Use .pdf or .docx.")

    resume_id = f"res_{uuid.uuid4().hex[:12]}"
    dest = UPLOAD_DIR / f"{resume_id}{ext}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    _resumes[resume_id] = dest
    return {"resumeId": resume_id, "filename": file.filename, "status": "received"}


@app.post("/job-descriptions", status_code=201)
async def upload_job_description(
    file: Optional[UploadFile] = File(None), text: Optional[str] = Form(None)
) -> dict:
    if not file and not text:
        raise HTTPException(400, "Provide either a file upload or raw `text`.")

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    pipeline = get_pipeline()

    if file:
        ext = Path(file.filename).suffix.lower() or ".txt"
        dest = UPLOAD_DIR / f"{job_id}{ext}"
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    else:
        dest = UPLOAD_DIR / f"{job_id}.txt"
        dest.write_text(text, encoding="utf-8")

    try:
        job_record = pipeline.process_job(dest)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Failed to parse job description: {exc}") from exc

    _jobs[job_id] = job_record
    _job_files[job_id] = dest
    return {
        "jobDescriptionId": job_id,
        "normalizedRole": job_record.get("normalized_role"),
        "requiredSkills": job_record.get("required_skills"),
        "status": "parsed",
    }


@app.post("/matches", status_code=201)
def create_match(request: MatchRequest) -> dict:
    resume_path = _resumes.get(request.resume_id)
    job_record = _jobs.get(request.job_description_id)
    if resume_path is None:
        raise HTTPException(404, f"Unknown resumeId '{request.resume_id}'.")
    if job_record is None:
        raise HTTPException(404, f"Unknown jobDescriptionId '{request.job_description_id}'.")

    pipeline = get_pipeline()
    run = pipeline.process_candidate(
        resume_path=resume_path,
        job_record=job_record,
        candidate_id=request.resume_id,
        job_id=request.job_description_id,
        role=request.role,
    )

    match_id = f"match_{uuid.uuid4().hex[:12]}"
    _matches[match_id] = run.to_dict()

    return {
        "matchId": match_id,
        "resumeId": request.resume_id,
        "jobDescriptionId": request.job_description_id,
        "finalScore": run.final_score,
        "matchBand": run.match_band,
        "status": run.status,
        "narrativeExplanation": run.narrative_explanation,
        "componentBreakdown": run.component_breakdown,
        "warnings": run.warnings,
    }


@app.get("/matches/{match_id}")
def get_match(match_id: str) -> dict:
    match = _matches.get(match_id)
    if match is None:
        raise HTTPException(404, f"Unknown matchId '{match_id}'.")
    return match


@app.get("/job-descriptions/{job_id}/shortlist")
def get_shortlist(job_id: str) -> dict:
    job_record = _jobs.get(job_id)
    if job_record is None:
        raise HTTPException(404, f"Unknown jobDescriptionId '{job_id}'.")

    resume_paths = list(_resumes.values())
    if not resume_paths:
        raise HTTPException(400, "No resumes have been uploaded yet.")

    pipeline = get_pipeline()
    result = pipeline.run_batch(
        resume_paths=resume_paths, job_path=_job_files[job_id], job_id=job_id
    )
    summary = result["summary"]
    return summary.to_dict()
