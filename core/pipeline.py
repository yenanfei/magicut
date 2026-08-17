"""
End-to-End Orchestrator Pipeline
Connects Target Tracker -> Human Detector -> Mask Processor -> Inpainter -> Video Encoder
"""

import os
import time
import cv2
import numpy as np
import yaml
import torch
from typing import List, Dict, Tuple, Optional, Callable

from .detector import HumanDetector
from .tracker import SAM2VideoTracker
from .mask_processor import DanceMaskProcessor
from .inpainter import VideoInpainter


class DancePersonRemoverPipeline:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)

        requested_device = self.config.get("system", {}).get("device", "cuda")
        if requested_device == "cuda" and not torch.cuda.is_available():
            print("[System] CUDA not available on this environment. Automatically falling back to CPU.")
            device = "cpu"
        else:
            device = requested_device

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

        # 2. Detect All Humans & Extract Instance Polygons (YOLO/Detector)
        if progress_cb:
            progress_cb(0.20, f"Segmenting all dancers & group instances across {total_frames} frames...")

        all_humans_masks, all_detections = self.detector.segment_video_frames(frames)

        # 3. Track Target Person with Exact Instance Contour
        if progress_cb:
            progress_cb(0.40, "Tracking target lead dancer with pixel-level precision...")

        self.tracker.init_video_state(video_path)
        target_masks = self.tracker.add_prompt_and_track(
            keyframe_idx=keyframe_idx,
            points=prompt_points,
            labels=prompt_labels,
            box=prompt_box,
            total_frames=total_frames,
            frame_shape=(h, w),
            video_detections=all_detections
        )

        # 4. Compute Removal Masks (Mask Subtraction + Floor Shadow Dilation)
        if progress_cb:
            progress_cb(0.50, "Calculating removal masks & stage shadow expansions...")

        removal_masks, meta_list = self.mask_processor.process_sequence(
            all_humans_masks, target_masks
        )

        # Sync inpainter config dynamically if changed
        inp_cfg = self.config.get("inpainter", {})
        self.inpainter.engine = inp_cfg.get("engine", self.inpainter.engine)
        self.inpainter.pcm_steps = inp_cfg.get("pcm_steps", self.inpainter.pcm_steps)

        # 5. Inpaint Video Background (ProPainter / DiffuEraser / Flow Engine)
        if progress_cb:
            progress_cb(0.70, f"Restoring stage background ({self.inpainter.engine})...")

        def inpaint_sub_cb(prog, txt):
            if progress_cb:
                progress_cb(0.70 + prog * 0.20, f"Inpainting: {txt}")

        clean_bg_frames = self.inpainter.inpaint_video(
            frames, removal_masks, all_humans_masks=all_humans_masks, progress_callback=inpaint_sub_cb
        )

        # 6. High-Fidelity Foreground Layer Recomposition
        # Preserves 100% of the lead protagonist's original crisp pixels, facial details, and hair
        if progress_cb:
            progress_cb(0.92, "Performing high-fidelity foreground alpha compositing...")

        final_solo_frames = []
        for i in range(len(frames)):
            orig_f = frames[i]
            bg_f = clean_bg_frames[i]
            if bg_f.shape[1] != w or bg_f.shape[0] != h:
                bg_f = cv2.resize(bg_f, (w, h), interpolation=cv2.INTER_LANCZOS4)

            rem_m = removal_masks[i]
            # Soft feathering on removal mask boundary for seamless background inpainting blend
            rem_mask_float = rem_m.astype(np.float32) / 255.0
            rem_alpha = cv2.GaussianBlur(rem_mask_float, (9, 9), 2.0)[:, :, None]

            # Clean stage background: 100% pristine original background outside removal + Inpainted stage inside removal
            clean_stage_bg = (bg_f.astype(np.float32) * rem_alpha + orig_f.astype(np.float32) * (1.0 - rem_alpha)).clip(0, 255).astype(np.uint8)

            t_mask = target_masks[i]
            if np.count_nonzero(t_mask) > 0:
                # Soft alpha matte with Gaussian smoothing around contours
                t_mask_float = t_mask.astype(np.float32) / 255.0
                alpha = cv2.GaussianBlur(t_mask_float, (7, 7), 1.5)[:, :, None]
                # Seamless blending: Original protagonist + Restored Clean Stage Background
                comp = (orig_f.astype(np.float32) * alpha + clean_stage_bg.astype(np.float32) * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
                final_solo_frames.append(comp)
            else:
                final_solo_frames.append(clean_stage_bg)

        # 7. Save Result Video
        if progress_cb:
            progress_cb(0.95, "Encoding clean solo video...")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        temp_no_audio = output_path.replace(".mp4", "_raw.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_no_audio, fourcc, fps, (w, h))

        for f in final_solo_frames:
            out.write(f)
        out.release()

        # Mux original audio if available
        try:
            import subprocess
            cmd = [
                "ffmpeg", "-y",
                "-i", temp_no_audio,
                "-i", video_path,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-shortest",
                output_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if os.path.exists(temp_no_audio):
                os.remove(temp_no_audio)
        except Exception:
            if os.path.exists(temp_no_audio):
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(temp_no_audio, output_path)

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

    def process_video(
        self,
        video_path: str,
        target_prompt: str = "center dancer",
        output_path: str = "outputs/solo_video.mp4",
        max_frames: Optional[int] = None,
        progress_cb: Optional[Callable[[float, str], None]] = None
    ) -> Dict:
        """
        High-level wrapper to process a video with text prompt describing the target person.
        """
        # Determine prompt coordinates from video dimensions & prompt description
        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
        cap.release()

        prompt_lower = target_prompt.lower()
        if "left" in prompt_lower:
            prompt_pts = np.array([[w * 0.25, h * 0.55]], dtype=np.float32)
        elif "right" in prompt_lower:
            prompt_pts = np.array([[w * 0.75, h * 0.55]], dtype=np.float32)
        else:  # default center dancer
            prompt_pts = np.array([[w * 0.50, h * 0.55]], dtype=np.float32)

        prompt_lbls = np.array([1], dtype=np.int32)

        return self.run(
            video_path=video_path,
            output_path=output_path,
            keyframe_idx=0,
            prompt_points=prompt_pts,
            prompt_labels=prompt_lbls,
            max_frames=max_frames,
            progress_cb=progress_cb
        )


# Alias for experimental interface compatibility
MagiCutPipeline = DancePersonRemoverPipeline

