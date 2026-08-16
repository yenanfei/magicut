"""
End-to-End Orchestrator Pipeline
Connects Target Tracker -> Human Detector -> Mask Processor -> Inpainter -> Video Encoder
"""

import os
import time
import cv2
import numpy as np
import yaml
from typing import List, Dict, Tuple, Optional, Callable

from .detector import HumanDetector
from .tracker import SAM2VideoTracker
from .mask_processor import DanceMaskProcessor
from .inpainter import VideoInpainter


class DancePersonRemoverPipeline:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)

        device = self.config.get("system", {}).get("device", "cuda")

        # Initialize sub-modules
        det_cfg = self.config.get("detector", {})
        self.detector = HumanDetector(
            model_name=det_cfg.get("model_name", "yolo11n-seg.pt"),
            conf_threshold=det_cfg.get("conf_threshold", 0.35),
            iou_threshold=det_cfg.get("iou_threshold", 0.6),
            device=device
        )

        sam_cfg = self.config.get("sam2", {})
        self.tracker = SAM2VideoTracker(
            model_cfg=sam_cfg.get("model_cfg", "sam2_hiera_l.yaml"),
            checkpoint_path=sam_cfg.get("checkpoint_path", "weights/sam2_hiera_large.pt"),
            device=device
        )

        mp_cfg = self.config.get("mask_processor", {})
        self.mask_processor = DanceMaskProcessor(
            general_dilation_kernel=mp_cfg.get("general_dilation_kernel", 7),
            shadow_dilation_y=mp_cfg.get("shadow_dilation_y", 25),
            shadow_dilation_x=mp_cfg.get("shadow_dilation_x", 9),
            feather_kernel=mp_cfg.get("feather_kernel", 5),
            occlusion_iou_thresh=mp_cfg.get("occlusion_iou_thresh", 0.15)
        )

        inp_cfg = self.config.get("inpainter", {})
        self.inpainter = VideoInpainter(
            engine=inp_cfg.get("engine", "propainter"),
            propainter_weights=inp_cfg.get("propainter_weights", "weights/ProPainter.pth"),
            subvideo_length=inp_cfg.get("subvideo_length", 80),
            neighbor_stride=inp_cfg.get("neighbor_stride", 10),
            device=device
        )

    def _load_config(self, path: str) -> dict:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def extract_frames_from_video(self, video_path: str, max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], float]:
        """
        Reads video and returns list of BGR frames and original FPS.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break
        cap.release()
        return frames, fps

    def run(
        self,
        video_path: str,
        output_path: str,
        keyframe_idx: int = 0,
        prompt_points: Optional[np.ndarray] = None, # [[x, y], ...]
        prompt_labels: Optional[np.ndarray] = None, # [1, 0, ...]
        prompt_box: Optional[np.ndarray] = None,    # [x1, y1, x2, y2]
        max_frames: Optional[int] = None,
        progress_cb: Optional[Callable[[float, str], None]] = None
    ) -> Dict:
        """
        Executes full dance person removal process.
        """
        start_time = time.time()
        if progress_cb:
            progress_cb(0.05, "Reading input video...")

        # 1. Load video
        frames, fps = self.extract_frames_from_video(video_path, max_frames=max_frames)
        total_frames = len(frames)
        if total_frames == 0:
            raise ValueError(f"Could not read video frames from {video_path}")

        h, w = frames[0].shape[:2]

        # 2. Track Target Person (SAM 2)
        if progress_cb:
            progress_cb(0.20, f"Tracking target member across {total_frames} frames...")

        self.tracker.init_video_state(video_path)
        target_masks = self.tracker.add_prompt_and_track(
            keyframe_idx=keyframe_idx,
            points=prompt_points,
            labels=prompt_labels,
            box=prompt_box,
            total_frames=total_frames,
            frame_shape=(h, w)
        )

        # 3. Detect All Humans (YOLO/Detector)
        if progress_cb:
            progress_cb(0.40, "Detecting all dancers & dance group formations...")

        all_humans_masks = self.detector.segment_video_frames(frames)

        # 4. Compute Removal Masks (Mask Subtraction + Floor Shadow Dilation)
        if progress_cb:
            progress_cb(0.55, "Calculating removal masks & shadow expansions...")

        removal_masks, meta_list = self.mask_processor.process_sequence(
            all_humans_masks, target_masks
        )

        # 5. Inpaint Video Background (ProPainter / Flow Engine)
        if progress_cb:
            progress_cb(0.70, "Restoring stage background (Video Inpainting)...")

        def inpaint_sub_cb(prog, txt):
            if progress_cb:
                progress_cb(0.70 + prog * 0.20, f"Inpainting: {txt}")

        clean_frames = self.inpainter.inpaint_video(
            frames, removal_masks, progress_callback=inpaint_sub_cb
        )

        # 6. Save Result Video
        if progress_cb:
            progress_cb(0.95, "Encoding clean solo video...")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for f in clean_frames:
            out.write(f)
        out.release()

        elapsed = time.time() - start_time
        if progress_cb:
            progress_cb(1.0, f"Completed successfully in {elapsed:.2f}s!")

        return {
            "output_path": output_path,
            "total_frames": total_frames,
            "fps": fps,
            "duration_sec": total_frames / fps,
            "elapsed_sec": elapsed,
            "meta": meta_list
        }
