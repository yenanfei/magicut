"""Pydantic schemas for the MagiCut cloud API."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    uploaded = "uploaded"
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class Point(BaseModel):
    x: float
    y: float


class ProcessRequest(BaseModel):
    keyframe_idx: int = Field(0, ge=0)
    points: List[Point] = Field(..., min_length=1)
    shadow_dilation: int = Field(25, ge=1, le=80)
    max_frames: Optional[int] = Field(
        None,
        description="Limit frames for prototype speed. null = use server default.",
    )


class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus
    filename: str
    message: str = "Video uploaded. Load a keyframe and select the target dancer."


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = 0.0
    stage: str = ""
    filename: Optional[str] = None
    error: Optional[str] = None
    result_ready: bool = False
    mode: Optional[str] = None
    elapsed_sec: Optional[float] = None
    total_frames: Optional[int] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "magicut-api"
    version: str
    pipeline_mode: str
    cuda_available: bool
