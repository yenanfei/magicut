"""
Human Instance Detector & Segmenter
Extracts binary segmentation masks of all dance members in each video frame.
"""

import numpy as np
import cv2
import torch
from typing import List, Dict, Tuple, Optional


class HumanDetector:
    def __init__(
        self,
        model_name: str = "yolo11n-seg.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.6,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model_name = model_name
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            from ultralytics import YOLO
            print(f"[Detector] Loading human segmentation model: {self.model_name} on {self.device}")
            self.model = YOLO(self.model_name)
        except Exception as e:
            print(f"[Detector] Warning: Could not initialize YOLO model ({e}). Will run in fallback mode.")
            self.model = None

    def segment_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Runs human instance segmentation on a single BGR frame.

        Returns:
            all_humans_mask: uint8 binary mask (H, W) where 255 represents any human body
            detections: List of dicts containing bbox [x1, y1, x2, y2], conf, and individual mask
        """
        h, w = frame_bgr.shape[:2]
        all_humans_mask = np.zeros((h, w), dtype=np.uint8)
        detections = []

        if self.model is None:
            # Fallback mock detector (e.g. for testing without heavy model weights)
            return all_humans_mask, detections

        results = self.model.predict(
            source=frame_bgr,
            classes=[0],  # 0 is 'person' in COCO dataset
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )

        if len(results) == 0 or results[0].masks is None:
            return all_humans_mask, detections

        res = results[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        raw_masks = res.masks.data.cpu().numpy()  # (N, H_mask, W_mask)

        for i, raw_mask in enumerate(raw_masks):
            # Resize mask to original frame size
            mask_resized = cv2.resize(
                raw_mask.astype(np.float32),
                (w, h),
                interpolation=cv2.INTER_LINEAR
            )
            mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

            all_humans_mask = np.bitwise_or(all_humans_mask, mask_binary)

            detections.append({
                "bbox": boxes[i].tolist(),
                "conf": float(confs[i]),
                "mask": mask_binary
            })

        return all_humans_mask, detections

    def segment_video_frames(self, frames_bgr: List[np.ndarray]) -> List[np.ndarray]:
        """
        Segments all frames in batch or sequence.
        """
        all_masks = []
        for frame in frames_bgr:
            mask, _ = self.segment_frame(frame)
            all_masks.append(mask)
        return all_masks
