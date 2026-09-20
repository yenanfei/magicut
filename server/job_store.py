"""Filesystem-backed job store for the MagiCut API prototype."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import JOBS_DIR, RESULT_DIR, UPLOAD_DIR, JOB_TTL_HOURS, ensure_dirs
from .schemas import JobStatus


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


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

    def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        ensure_dirs()
        items: List[Dict[str, Any]] = []
        with self._lock:
            for path in JOBS_DIR.glob("*.json"):
                try:
                    items.append(json.loads(path.read_text(encoding="utf-8")))
                except Exception:
                    continue
        items.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        return items[: max(1, limit)]

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

    def delete(self, job_id: str) -> bool:
        with self._lock:
            record_path = self._path(job_id)
            if not record_path.exists():
                return False
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
            except Exception:
                record = {}
            for key in ("upload_path", "result_path"):
                p = Path(record.get(key) or "")
                if p.exists() and p.is_file():
                    p.unlink(missing_ok=True)
            record_path.unlink(missing_ok=True)
            return True

    def cleanup_expired(self, ttl_hours: Optional[float] = None) -> int:
        ttl = JOB_TTL_HOURS if ttl_hours is None else ttl_hours
        if ttl <= 0:
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(hours=ttl)
        removed = 0
        for job in self.list_jobs(limit=10_000):
            created = _parse_ts(job.get("created_at"))
            if created and created < cutoff:
                if self.delete(job["job_id"]):
                    removed += 1
        return removed

    def _write(self, record: Dict[str, Any]) -> None:
        path = self._path(record["job_id"])
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
        tmp.replace(path)


def upload_destination(job_id: str, filename: str) -> Path:
    ensure_dirs()
    suffix = Path(filename).suffix.lower() or ".mp4"
    return UPLOAD_DIR / f"{job_id}{suffix}"
