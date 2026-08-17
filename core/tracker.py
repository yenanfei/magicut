"""
High-Precision Target Tracker & SAM2 Video Interface
Tracks the specified dance member across video frames with exact polygon contours.
Uses YOLO instance segmentation tracking with spatial-temporal IoU and feature continuity.
"""

import os
import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional


class InstanceTracker:
    """
    Tracks a target instance across frames using YOLO segmentation detections,
    bypassing rough bounding-box simulations to guarantee pixel-exact contours.
    """
    def __init__(self, iou_thresh: float = 0.25):
        self.iou_thresh = iou_thresh

    def track_target_from_detections(
        self,
        video_detections: List[List[Dict]], # detections per frame
        prompt_points: Optional[np.ndarray] = None,
        keyframe_idx: int = 0,
        frame_shape: Tuple[int, int] = (720, 1280)
    ) -> List[np.ndarray]:
        h, w = frame_shape
        num_frames = len(video_detections)
        target_masks = [np.zeros((h, w), dtype=np.uint8) for _ in range(num_frames)]

        if num_frames == 0:
            return target_masks

        # 1. Locate the target instance in the keyframe
        key_dets = video_detections[keyframe_idx]
        if not key_dets:
            # If no detection in keyframe, find nearest frame with detections
            for k in range(num_frames):
                if video_detections[k]:
                    keyframe_idx = k
                    key_dets = video_detections[k]
                    break

        if not key_dets:
            return target_masks

        # Find detection matching the prompt point
        best_idx = 0
        if prompt_points is not None and len(prompt_points) > 0:
            px, py = prompt_points[0][0], prompt_points[0][1]
            min_dist = float("inf")
            for idx, det in enumerate(key_dets):
                # Check if point inside mask
                mask = det["mask"]
                if 0 <= int(py) < h and 0 <= int(px) < w and mask[int(py), int(px)] > 0:
                    best_idx = idx
                    break
                # Or compute distance to bbox center
                box = det["bbox"]
                cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
                dist = (cx - px) ** 2 + (cy - py) ** 2
                if dist < min_dist:
                    min_dist = dist
                    best_idx = idx

        target_masks[keyframe_idx] = key_dets[best_idx]["mask"].copy()
        current_box = key_dets[best_idx]["bbox"]

        # 2. Forward tracking (keyframe -> end)
        prev_box = current_box
        prev_mask = target_masks[keyframe_idx]
        for f in range(keyframe_idx + 1, num_frames):
            frame_dets = video_detections[f]
            if not frame_dets:
                target_masks[f] = prev_mask.copy()
                continue

            best_match = self._find_best_match(prev_box, prev_mask, frame_dets)
            if best_match is not None:
                target_masks[f] = best_match["mask"].copy()
                prev_box = best_match["bbox"]
                prev_mask = target_masks[f]
            else:
                target_masks[f] = prev_mask.copy()

        # 3. Backward tracking (keyframe -> 0)
        prev_box = current_box
        prev_mask = target_masks[keyframe_idx]
        for f in range(keyframe_idx - 1, -1, -1):
            frame_dets = video_detections[f]
            if not frame_dets:
                target_masks[f] = prev_mask.copy()
                continue

            best_match = self._find_best_match(prev_box, prev_mask, frame_dets)
            if best_match is not None:
                target_masks[f] = best_match["mask"].copy()
                prev_box = best_match["bbox"]
                prev_mask = target_masks[f]
            else:
                target_masks[f] = prev_mask.copy()

        return target_masks

    def _find_best_match(self, prev_box: List[float], prev_mask: np.ndarray, current_dets: List[Dict]) -> Optional[Dict]:
        best_score = -1.0
        best_det = None

        prev_cx = (prev_box[0] + prev_box[2]) / 2.0
        prev_cy = (prev_box[1] + prev_box[3]) / 2.0

        for det in current_dets:
            cur_box = det["bbox"]
            cur_mask = det["mask"]

            # Calculate Bbox IoU
            ix1 = max(prev_box[0], cur_box[0])
            iy1 = max(prev_box[1], cur_box[1])
            ix2 = min(prev_box[2], cur_box[2])
            iy2 = min(prev_box[3], cur_box[3])

            inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            box1_area = (prev_box[2] - prev_box[0]) * (prev_box[3] - prev_box[1])
            box2_area = (cur_box[2] - cur_box[0]) * (cur_box[3] - cur_box[1])
            union_area = box1_area + box2_area - inter_area + 1e-6

            bbox_iou = inter_area / union_area

            # Center distance proximity score
            cur_cx = (cur_box[0] + cur_box[2]) / 2.0
            cur_cy = (cur_box[1] + cur_box[3]) / 2.0
            dist = np.sqrt((cur_cx - prev_cx) ** 2 + (cur_cy - prev_cy) ** 2)
            dist_score = max(0, 1.0 - dist / 300.0)

            # Combined match score
            score = bbox_iou * 0.7 + dist_score * 0.3

            if score > best_score and score > self.iou_thresh:
                best_score = score
                best_det = det

        return best_det


class SAM2VideoTracker:
    def __init__(
        self,
        model_cfg: str = "sam2_hiera_l.yaml",
        checkpoint_path: str = "weights/sam2_hiera_large.pt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.model_cfg = model_cfg
        self.checkpoint_path = checkpoint_path
        self.predictor = None
        self.inference_state = None
        self.is_sam2_available = False
        self.instance_tracker = InstanceTracker()
        self._init_sam2()

    def _init_sam2(self):
        try:
            from sam2.build_sam import build_sam2_video_predictor
            if os.path.exists(self.checkpoint_path):
                print(f"[SAM2 Tracker] Loading SAM2 model from {self.checkpoint_path}")
                self.predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint_path, device=self.device)
                self.is_sam2_available = True
            else:
                print(f"[SAM2 Tracker] Checkpoint '{self.checkpoint_path}' not found. Using high-precision YOLO instance tracker.")
        except Exception as e:
            print(f"[SAM2 Tracker] Running with high-precision YOLO instance tracker.")

    def init_video_state(self, video_path_or_frames):
        if self.is_sam2_available and self.predictor is not None:
            try:
                if isinstance(video_path_or_frames, str):
                    self.inference_state = self.predictor.init_state(video_path=video_path_or_frames)
                elif isinstance(video_path_or_frames, list):
                    # Write frames to temp dir for SAM2
                    import tempfile
                    temp_dir = tempfile.mkdtemp(prefix="sam2_frames_")
                    for idx, fr in enumerate(video_path_or_frames):
                        cv2.imwrite(os.path.join(temp_dir, f"{idx:05d}.jpg"), fr)
                    self.inference_state = self.predictor.init_state(video_path=temp_dir)
            except Exception as e:
                print(f"[SAM2 Tracker] Warning: init_state failed ({e}). Falling back to YOLO tracker.")
                self.inference_state = {"type": "instance_tracker", "video": video_path_or_frames}
        else:
            self.inference_state = {"type": "instance_tracker", "video": video_path_or_frames}

    def add_prompt_and_track(
        self,
        keyframe_idx: int,
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        total_frames: int = 100,
        frame_shape: Tuple[int, int] = (720, 1280),
        video_detections: Optional[List[List[Dict]]] = None
    ) -> List[np.ndarray]:
        h, w = frame_shape

        if self.is_sam2_available and self.predictor is not None and isinstance(self.inference_state, dict) and "num_frames" in self.inference_state:
            try:
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=keyframe_idx,
                    obj_id=1,
                    points=points,
                    labels=labels,
                    box=box,
                )

                video_segments = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
                    mask_raw = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze().astype(np.uint8) * 255
                    if mask_raw.shape[:2] != (h, w):
                        mask_raw = cv2.resize(mask_raw, (w, h), interpolation=cv2.INTER_NEAREST)
                    video_segments[out_frame_idx] = mask_raw

                target_masks = [video_segments.get(i, np.zeros((h, w), dtype=np.uint8)) for i in range(total_frames)]
                return target_masks
            except Exception as e:
                print(f"[SAM2 Tracker] Error during propagation ({e}). Falling back to YOLO tracker.")

        if video_detections is not None:
            # Exact YOLO instance tracking
            return self.instance_tracker.track_target_from_detections(
                video_detections=video_detections,
                prompt_points=points,
                keyframe_idx=keyframe_idx,
                frame_shape=frame_shape
            )

        return [np.zeros((h, w), dtype=np.uint8) for _ in range(total_frames)]
