#!/usr/bin/env python3
"""End-to-end smoke test against a running MagiCut API."""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import cv2
import httpx
import numpy as np


def make_sample_video(path: Path, frames: int = 24, w: int = 320, h: int = 180) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        12.0,
        (w, h),
    )
    for i in range(frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = (30, 40, 50)
        # three "dancers"
        for idx, cx in enumerate((80, 160, 240)):
            color = (40, 180, 90) if idx == 1 else (90, 90, 200)
            cv2.circle(frame, (cx, 90 + (i % 5)), 28, color, -1)
        writer.write(frame)
    writer.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8080")
    parser.add_argument("--token", default="")
    args = parser.parse_args()
    headers = {}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"

    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "sample.mp4"
        make_sample_video(video)

        with httpx.Client(base_url=args.base, headers=headers, timeout=120.0) as client:
            health = client.get("/health")
            health.raise_for_status()
            print("health:", health.json())

            demo = client.post("/api/v1/jobs/demo")
            demo.raise_for_status()
            print("demo job:", demo.json()["job_id"])

            listed = client.get("/api/v1/jobs", params={"limit": 5})
            listed.raise_for_status()
            print("list count:", listed.json()["count"])

            with video.open("rb") as f:
                created = client.post(
                    "/api/v1/jobs",
                    files={"video": ("sample.mp4", f, "video/mp4")},
                )
            created.raise_for_status()
            job_id = created.json()["job_id"]
            print("job:", job_id)

            kf = client.get(f"/api/v1/jobs/{job_id}/keyframe", params={"frame": 0})
            kf.raise_for_status()
            assert kf.headers["content-type"].startswith("image/")
            print("keyframe bytes:", len(kf.content))

            proc = client.post(
                f"/api/v1/jobs/{job_id}/process",
                json={
                    "keyframe_idx": 0,
                    "points": [{"x": 160, "y": 90}],
                    "shadow_dilation": 25,
                    "max_frames": 20,
                },
            )
            proc.raise_for_status()
            print("queued:", proc.json()["status"])

            for _ in range(120):
                st = client.get(f"/api/v1/jobs/{job_id}")
                st.raise_for_status()
                body = st.json()
                print(f"  {body['status']} {body['progress']:.2f} {body['stage']}")
                if body["status"] in {"completed", "failed"}:
                    break
                time.sleep(0.5)
            else:
                print("TIMEOUT waiting for job", file=sys.stderr)
                return 1

            if body["status"] != "completed" or not body["result_ready"]:
                print("FAILED:", body, file=sys.stderr)
                return 1

            result = client.get(f"/api/v1/jobs/{job_id}/result")
            result.raise_for_status()
            out = Path(tmp) / "result.mp4"
            out.write_bytes(result.content)
            print("result bytes:", out.stat().st_size, "mode:", body.get("mode"))
            if out.stat().st_size < 1000:
                print("Result too small", file=sys.stderr)
                return 1

    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
