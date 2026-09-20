"""Pipeline runners: real MagiCut GPU path + lightweight mock for prototype demos."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import CONFIG_PATH, PIPELINE_MODE, ROOT_DIR


ProgressCB = Callable[[float, str], None]


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_mode(requested: Optional[str] = None) -> str:
    mode = (requested or PIPELINE_MODE or "auto").lower()
    if mode == "auto":
        if cuda_available() and (ROOT_DIR / "weights" / "sam2_hiera_large.pt").exists():
            return "real"
        return "mock"
    if mode not in {"mock", "real"}:
        return "mock"
    return mode


def extract_keyframe_jpeg(video_path: str, frame_index: int) -> Tuple[bytes, int, int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open uploaded video")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frame_index = min(max(0, frame_index), max(0, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError(f"Failed to read frame {frame_index}")
    h, w = frame.shape[:2]
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise ValueError("Failed to encode keyframe")
    return buf.tobytes(), frame_index, total, w


def _mux_audio(raw_video: Path, source_video: Path, output_path: Path) -> None:
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_video),
            "-i",
            str(source_video),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-shortest",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        raw_video.unlink(missing_ok=True)
    except Exception:
        if output_path.exists():
            output_path.unlink()
        raw_video.replace(output_path)


def run_mock_pipeline(
    video_path: str,
    output_path: str,
    keyframe_idx: int,
    points: List[Tuple[float, float]],
    max_frames: Optional[int],
    progress_cb: Optional[ProgressCB] = None,
) -> Dict:
    """Prototype runner that works without GPU/weights.

    Keeps a soft focus around the selected dancer point and dims the rest,
    enough to exercise the full mobile → API → result loop.
    """
    start = time.time()
    if progress_cb:
        progress_cb(0.05, "Reading input video (mock)...")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames: List[np.ndarray] = []
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise ValueError("Could not read video frames")

    h, w = frames[0].shape[:2]
    px = float(np.clip(points[0][0], 0, w - 1))
    py = float(np.clip(points[0][1], 0, h - 1))
    # Focus radius ~35% of the shorter side, scaled for portrait/landscape
    radius = max(40.0, 0.35 * min(h, w))

    ys, xs = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xs - px) ** 2 + (ys - py) ** 2)
    # Soft keep-mask around selected point
    keep = np.clip(1.0 - (dist - radius * 0.55) / (radius * 0.75), 0.0, 1.0)
    keep = keep.astype(np.float32)[:, :, None]

    out_frames: List[np.ndarray] = []
    total = len(frames)
    for i, frame in enumerate(frames):
        if progress_cb and (i % max(1, total // 10) == 0):
            progress_cb(0.1 + 0.75 * (i / total), f"Mock solo render {i + 1}/{total}...")

        blurred = cv2.GaussianBlur(frame, (31, 31), 0)
        dimmed = (blurred.astype(np.float32) * 0.35).astype(np.uint8)
        comp = (frame.astype(np.float32) * keep + dimmed.astype(np.float32) * (1.0 - keep))
        comp = np.clip(comp, 0, 255).astype(np.uint8)

        # Subtle brand watermark for prototype clarity
        cv2.putText(
            comp,
            "MagiCut Prototype",
            (16, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        out_frames.append(comp)
        # Tiny sleep so progress polling is visible in demos
        time.sleep(0.01)

    if progress_cb:
        progress_cb(0.92, "Encoding output video...")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_name(out.stem + "_raw.mp4")
    writer = cv2.VideoWriter(
        str(raw),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
    )
    for f in out_frames:
        writer.write(f)
    writer.release()
    _mux_audio(raw, Path(video_path), out)

    elapsed = time.time() - start
    if progress_cb:
        progress_cb(1.0, f"Completed (mock) in {elapsed:.1f}s")
    return {
        "output_path": str(out),
        "total_frames": total,
        "fps": float(fps),
        "duration_sec": total / float(fps),
        "elapsed_sec": elapsed,
        "mode": "mock",
    }


def run_real_pipeline(
    video_path: str,
    output_path: str,
    keyframe_idx: int,
    points: List[Tuple[float, float]],
    shadow_dilation: int,
    max_frames: Optional[int],
    progress_cb: Optional[ProgressCB] = None,
) -> Dict:
    # Ensure repo root is importable
    import sys

    root = str(ROOT_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)

    from core.pipeline import DancePersonRemoverPipeline

    pipe = DancePersonRemoverPipeline(config_path=str(CONFIG_PATH))
    pipe.mask_processor.shadow_dilation_y = int(shadow_dilation)
    pts = np.array(points, dtype=np.float32)
    labels = np.ones(len(points), dtype=np.int32)
    result = pipe.run(
        video_path=video_path,
        output_path=output_path,
        keyframe_idx=int(keyframe_idx),
        prompt_points=pts,
        prompt_labels=labels,
        max_frames=max_frames,
        progress_cb=progress_cb,
    )
    result["mode"] = "real"
    return result


def run_pipeline(
    video_path: str,
    output_path: str,
    keyframe_idx: int,
    points: List[Tuple[float, float]],
    shadow_dilation: int = 25,
    max_frames: Optional[int] = None,
    progress_cb: Optional[ProgressCB] = None,
    mode: Optional[str] = None,
) -> Dict:
    chosen = resolve_mode(mode)
    if chosen == "real":
        try:
            return run_real_pipeline(
                video_path=video_path,
                output_path=output_path,
                keyframe_idx=keyframe_idx,
                points=points,
                shadow_dilation=shadow_dilation,
                max_frames=max_frames,
                progress_cb=progress_cb,
            )
        except Exception as exc:
            # Fall back so mobile prototype still works on CPU-only hosts
            if progress_cb:
                progress_cb(0.02, f"Real pipeline unavailable ({exc}); falling back to mock...")
            result = run_mock_pipeline(
                video_path=video_path,
                output_path=output_path,
                keyframe_idx=keyframe_idx,
                points=points,
                max_frames=max_frames,
                progress_cb=progress_cb,
            )
            result["mode"] = "mock_fallback"
            result["fallback_reason"] = str(exc)
            return result

    return run_mock_pipeline(
        video_path=video_path,
        output_path=output_path,
        keyframe_idx=keyframe_idx,
        points=points,
        max_frames=max_frames,
        progress_cb=progress_cb,
    )
