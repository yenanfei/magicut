"""
Mask Disentanglement, Directional Shadow Dilation, and Occlusion Processing
Specialized algorithms tailored for dance videos to eliminate ghost shadows and boundary halos.
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Optional, List


class DanceMaskProcessor:
    def __init__(
        self,
        general_dilation_kernel: int = 7,
        shadow_dilation_y: int = 12,
        shadow_dilation_x: int = 5,
        feather_kernel: int = 5,
        occlusion_iou_thresh: float = 0.15
    ):
        self.general_dilation = general_dilation_kernel
        self.shadow_dilation_y = shadow_dilation_y
        self.shadow_dilation_x = shadow_dilation_x
        self.feather_kernel = feather_kernel
        self.occlusion_iou_thresh = occlusion_iou_thresh

        # Standard circular / elliptical kernel for general boundary dilation
        self.general_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.general_dilation, self.general_dilation)
        )

        # Directional foot shadow dilation kernel (downwards only)
        self.shadow_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.shadow_dilation_x, self.shadow_dilation_y)
        )

    def compute_removal_mask(
        self,
        all_humans_mask: np.ndarray,
        target_mask: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Calculates the precise area to be erased (Removal Mask).
        
        Formula:
            Other_Humans = All_Humans AND (NOT Target)
            Expanded_Removal = DirectionalDilation(Other_Humans)
            Clean_Removal = Expanded_Removal AND (NOT Protected_Target)
        """
        h, w = target_mask.shape[:2]

        # 1. Base subtraction: Other dancers = All humans minus Target
        other_humans_raw = cv2.bitwise_and(
            all_humans_mask, cv2.bitwise_not(target_mask)
        )

        # 2. Check for occlusion intersection between target and others
        intersection = cv2.bitwise_and(all_humans_mask, target_mask)
        target_area = np.count_nonzero(target_mask)
        intersect_area = np.count_nonzero(intersection)

        overlap_ratio = intersect_area / max(1, target_area)
        is_occluded = overlap_ratio > self.occlusion_iou_thresh

        # 3. Apply controlled general dilation to cover clothing fringes and hair
        other_dilated = cv2.dilate(other_humans_raw, self.general_kernel, iterations=1)

        # 4. Apply subtle downwards foot shadow dilation
        shadow_dilated = cv2.dilate(other_dilated, self.shadow_kernel, iterations=1)

        # 5. Strict Target Protection: Ensure the target protagonist is NEVER erased
        target_protective_margin = cv2.dilate(
            target_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1
        )
        final_removal_mask = cv2.bitwise_and(
            shadow_dilated, cv2.bitwise_not(target_protective_margin)
        )

        # 6. Feathering / Edge smoothing
        if self.feather_kernel > 1 and self.feather_kernel % 2 == 1:
            soft_mask = cv2.GaussianBlur(
                final_removal_mask, (self.feather_kernel, self.feather_kernel), 0
            )
            final_removal_mask = (soft_mask > 20).astype(np.uint8) * 255

        meta_info = {
            "is_occluded": bool(is_occluded),
            "overlap_ratio": float(overlap_ratio),
            "erased_pixel_count": int(np.count_nonzero(final_removal_mask)),
            "target_pixel_count": int(target_area)
        }

        return final_removal_mask, meta_info

    def process_sequence(
        self,
        all_humans_masks: List[np.ndarray],
        target_masks: List[np.ndarray],
        temporal_window_radius: int = 2
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Processes a full sequence of masks with multi-frame temporal continuity smoothing.
        Eliminates single-frame mask dropouts and flickering during fast dancer turns.
        """
        raw_removal_masks = []
        meta_list = []

        # 1. Compute per-frame baseline removal masks
        for i, (all_m, tgt_m) in enumerate(zip(all_humans_masks, target_masks)):
            rem_m, meta = self.compute_removal_mask(all_m, tgt_m)
            raw_removal_masks.append(rem_m)
            meta_list.append(meta)

        # 2. Multi-frame Temporal Smoothing along time axis
        num_frames = len(raw_removal_masks)
        final_removal_masks = []

        for t in range(num_frames):
            t_min = max(0, t - temporal_window_radius)
            t_max = min(num_frames, t + temporal_window_radius + 1)

            # Temporal union across neighboring frames
            temporal_accum = np.zeros_like(raw_removal_masks[0])
            for k in range(t_min, t_max):
                temporal_accum = np.maximum(temporal_accum, raw_removal_masks[k])

            # Strict protection: Subtract protagonist with safety margin
            target_protected = cv2.dilate(
                target_masks[t],
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1
            )
            smoothed_rem_mask = cv2.bitwise_and(
                temporal_accum, cv2.bitwise_not(target_protected)
            )
            final_removal_masks.append(smoothed_rem_mask)

        return final_removal_masks, meta_list
