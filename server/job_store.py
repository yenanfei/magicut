"""Filesystem-backed job store for the MagiCut API prototype."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .config import JOBS_DIR, RESULT_DIR, UPLOAD_DIR, ensure_dirs
from .schemas import JobStatus


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self) -> None:
        ensure_dirs()
        self._lock = threading.Lock()

    def _path(self, job_id: str) -> Path:
        return JOBS_DIR / f"{job_id}.json"

    def create(
        self,
        filename: str,
        upload_path: Path,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        job_id = job_id or uuid.uuid4().hex[:12]
        record = {
            "job_id": job_id,
            "status": JobStatus.uploaded.value,
            "progress": 0.0,
            "stage": "uploaded",
            "filename": filename,
            "upload_path": str(upload_path),
            "result_path": str(RESULT_DIR / f"{job_id}.mp4"),
            "error": None,
            "mode": None,
            "elapsed_sec": None,
            "total_frames": None,
            "created_at": _now(),
            "updated_at": _now(),
            "params": None,
        }
        with self._lock:
            self._write(record)
        return record

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        path = self._path(job_id)
        if not path.exists():
            return None
        with self._lock:
            return json.loads(path.read_text(encoding="utf-8"))

    def update(self, job_id: str, **fields: Any) -> Dict[str, Any]:
        with self._lock:
            record = json.loads(self._path(job_id).read_text(encoding="utf-8"))
            record.update(fields)
            record["updated_at"] = _now()
            self._write(record)
            return record

    def set_progress(self, job_id: str, progress: float, stage: str) -> None:
        self.update(
            job_id,
            progress=float(max(0.0, min(1.0, progress))),
            stage=stage,
            status=JobStatus.processing.value,
        )

    def _write(self, record: Dict[str, Any]) -> None:
        path = self._path(record["job_id"])
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
        tmp.replace(path)


def upload_destination(job_id: str, filename: str) -> Path:
    ensure_dirs()
    suffix = Path(filename).suffix.lower() or ".mp4"
    return UPLOAD_DIR / f"{job_id}{suffix}"
