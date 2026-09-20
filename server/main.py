"""MagiCut FastAPI server — mobile / App Store backend prototype."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import (
    CORS_ORIGINS,
    MAX_UPLOAD_MB,
    ROOT_DIR,
    API_TOKEN,
    ensure_dirs,
)
from .job_store import JobStore, upload_destination
from .pipeline_runner import cuda_available, extract_keyframe_jpeg, resolve_mode
from .schemas import (
    HealthResponse,
    JobCreateResponse,
    JobStatus,
    JobStatusResponse,
    ProcessRequest,
)
from .worker import JobWorker

ensure_dirs()
store = JobStore()
worker = JobWorker(store)
worker.start()

app = FastAPI(
    title="MagiCut API",
    description="Cloud GPU backend prototype for MagiCut iOS / mobile clients.",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MOBILE_WEB = ROOT_DIR / "clients" / "mobile_web"
if MOBILE_WEB.exists():
    app.mount("/app", StaticFiles(directory=str(MOBILE_WEB), html=True), name="mobile_web")


def require_token(authorization: Optional[str] = Header(default=None)) -> None:
    if not API_TOKEN:
        return
    if not authorization or authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid or missing API token")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        version=__version__,
        pipeline_mode=resolve_mode(),
        cuda_available=cuda_available(),
    )


@app.post("/api/v1/jobs", response_model=JobCreateResponse, dependencies=[Depends(require_token)])
async def create_job(video: UploadFile = File(...)) -> JobCreateResponse:
    if not video.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    suffix = Path(video.filename).suffix.lower()
    if suffix not in {".mp4", ".mov", ".m4v", ".avi", ".webm"}:
        raise HTTPException(status_code=400, detail=f"Unsupported video type: {suffix}")

    job_id = uuid4().hex[:12]
    dest = upload_destination(job_id, video.filename)
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    try:
        with dest.open("wb") as f:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    dest.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds {MAX_UPLOAD_MB}MB limit",
                    )
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc

    record = store.create(filename=video.filename, upload_path=dest, job_id=job_id)
    return JobCreateResponse(
        job_id=record["job_id"],
        status=JobStatus.uploaded,
        filename=video.filename,
    )


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse, dependencies=[Depends(require_token)])
def get_job(job_id: str) -> JobStatusResponse:
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job_id,
        status=JobStatus(job["status"]),
        progress=float(job.get("progress") or 0),
        stage=job.get("stage") or "",
        filename=job.get("filename"),
        error=job.get("error"),
        result_ready=job.get("status") == JobStatus.completed.value
        and Path(job.get("result_path", "")).exists(),
        mode=job.get("mode"),
        elapsed_sec=job.get("elapsed_sec"),
        total_frames=job.get("total_frames"),
    )


@app.get("/api/v1/jobs/{job_id}/keyframe", dependencies=[Depends(require_token)])
def get_keyframe(job_id: str, frame: int = Query(0, ge=0)) -> Response:
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    try:
        jpeg, used_idx, total, width = extract_keyframe_jpeg(job["upload_path"], frame)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={
            "X-Frame-Index": str(used_idx),
            "X-Frame-Total": str(total),
            "X-Frame-Width": str(width),
        },
    )


@app.post("/api/v1/jobs/{job_id}/process", response_model=JobStatusResponse, dependencies=[Depends(require_token)])
def process_job(job_id: str, body: ProcessRequest) -> JobStatusResponse:
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] in {JobStatus.queued.value, JobStatus.processing.value}:
        raise HTTPException(status_code=409, detail="Job already running")

    store.update(
        job_id,
        params={
            "keyframe_idx": body.keyframe_idx,
            "points": [p.model_dump() for p in body.points],
            "shadow_dilation": body.shadow_dilation,
            "max_frames": body.max_frames,
        },
        error=None,
        mode=None,
        elapsed_sec=None,
        total_frames=None,
    )
    worker.enqueue(job_id)
    job = store.get(job_id)
    return JobStatusResponse(
        job_id=job_id,
        status=JobStatus(job["status"]),
        progress=float(job.get("progress") or 0),
        stage=job.get("stage") or "",
        filename=job.get("filename"),
        result_ready=False,
    )


@app.get("/api/v1/jobs/{job_id}/result", dependencies=[Depends(require_token)])
def download_result(job_id: str):
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != JobStatus.completed.value:
        raise HTTPException(status_code=409, detail=f"Job not completed (status={job['status']})")
    path = Path(job["result_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result file missing")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"magicut_{job_id}.mp4",
    )


@app.get("/")
def root():
    return {
        "service": "magicut-api",
        "docs": "/docs",
        "mobile_web": "/app/",
        "health": "/health",
    }
