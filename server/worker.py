"""Background worker that executes MagiCut jobs one at a time."""

from __future__ import annotations

import queue
import threading
import traceback
from typing import Any, Dict, Optional, Tuple

from .config import DEFAULT_MAX_FRAMES
from .job_store import JobStore
from .pipeline_runner import run_pipeline
from .schemas import JobStatus


class JobWorker:
    def __init__(self, store: JobStore) -> None:
        self.store = store
        self._q: "queue.Queue[str]" = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="magicut-worker", daemon=True)
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()

    def enqueue(self, job_id: str) -> None:
        self.store.update(job_id, status=JobStatus.queued.value, stage="queued", progress=0.0)
        self._q.put(job_id)

    def _loop(self) -> None:
        while True:
            job_id = self._q.get()
            try:
                self._process(job_id)
            except Exception:
                traceback.print_exc()
                self.store.update(
                    job_id,
                    status=JobStatus.failed.value,
                    stage="failed",
                    error="Unhandled worker exception",
                    progress=1.0,
                )
            finally:
                self._q.task_done()

    def _process(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if not job:
            return
        params: Dict[str, Any] = job.get("params") or {}
        points = [(float(p["x"]), float(p["y"])) for p in params.get("points", [])]
        if not points:
            self.store.update(
                job_id,
                status=JobStatus.failed.value,
                error="No target points provided",
                stage="failed",
            )
            return

        max_frames = params.get("max_frames")
        if max_frames is None:
            max_frames = DEFAULT_MAX_FRAMES
        if max_frames == 0:
            max_frames = None

        def progress_cb(frac: float, desc: str) -> None:
            self.store.set_progress(job_id, frac, desc)

        self.store.update(job_id, status=JobStatus.processing.value, stage="starting", error=None)
        try:
            result = run_pipeline(
                video_path=job["upload_path"],
                output_path=job["result_path"],
                keyframe_idx=int(params.get("keyframe_idx", 0)),
                points=points,
                shadow_dilation=int(params.get("shadow_dilation", 25)),
                max_frames=max_frames,
                progress_cb=progress_cb,
            )
            self.store.update(
                job_id,
                status=JobStatus.completed.value,
                progress=1.0,
                stage="completed",
                mode=result.get("mode"),
                elapsed_sec=result.get("elapsed_sec"),
                total_frames=result.get("total_frames"),
                error=None,
            )
        except Exception as exc:
            traceback.print_exc()
            self.store.update(
                job_id,
                status=JobStatus.failed.value,
                stage="failed",
                error=str(exc),
                progress=1.0,
            )
