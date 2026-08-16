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
        shadow_dilation_y: int = 25,
        shadow_dilation_x: int = 9,
        feather_kernel: int = 5,
        occlusion_iou_thresh: float = 0.15
    ):
        self.general_dilation = general_dilation_kernel
        self.shadow_dilation_y = shadow_dilation_y
        self.shadow_dilation_x = shadow_dilation_x
        self.feather_kernel = feather_kernel
        self.occlusion_iou_thresh = occlusion_iou_thresh

        # Standard circular / elliptical kernel for general dilation
        self.general_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.general_dilation, self.general_dilation)
        )
        
        # Asymmetrical directional kernel: extends heavily downward towards floor for dance stage shadows
        self.shadow_kernel = self._create_directional_shadow_kernel(
            self.shadow_dilation_x, self.shadow_dilation_y
        )

    def _create_directional_shadow_kernel(self, kx: int, ky: int) -> np.ndarray:
        """
        Creates a kernel with anchor at the top center, expanding downwards towards the stage floor.
        """
        kernel = np.zeros((ky, kx), dtype=np.uint8)
        # Fill a cone/ellipse extending downwards
        center_x = kx // 2
        for y in range(ky):
            # Spread widens slightly as it goes down
            spread = int((center_x) * (y / max(1, ky - 1)))
            x_min = max(0, center_x - spread)
            x_max = min(kx, center_x + spread + 1)
            kernel[y, x_min:x_max] = 1
        return kernel

    def compute_removal_mask(
        self,
        all_humans_mask: np.ndarray,
        target_mask: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Calculates the exact area to be erased (Removal Mask).
        
        Formula:
            Other_Humans = All_Humans AND (NOT Target)
            Expanded_Removal = DirectionalDilation(Other_Humans)
            Clean_Removal = Expanded_Removal AND (NOT Protected_Target)

        Returns:
            final_removal_mask: uint8 (H, W) where 255 represents pixels to erase
            meta_info: Dictionary containing occlusion status and statistics
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

        # 3. Apply general dilation to cover clothing fringes and hair motion blur
        other_dilated = cv2.dilate(other_humans_raw, self.general_kernel, iterations=1)

        # 4. Apply directional floor shadow dilation to capture stage lighting shadows
        shadow_expanded = cv2.filter2D(
            other_dilated.astype(np.float32), -1, self.shadow_kernel
        )
        shadow_mask = (shadow_expanded > 0).astype(np.uint8) * 255

        # Combined removal candidate
        removal_candidate = cv2.bitwise_or(other_dilated, shadow_mask)

        # 5. Strict Target Protection: Ensure the target person is NEVER erased
        # Protect target with a slight margin
        target_protective_margin = cv2.dilate(
            target_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1
        )
        final_removal_mask = cv2.bitwise_and(
            removal_candidate, cv2.bitwise_not(target_protective_margin)
        )

        # 6. Feathering / Edge smoothing
        if self.feather_kernel > 1:
            # Gaussian blur for soft transition
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
        target_masks: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Processes a full sequence of masks with temporal continuity check.
        """
        removal_masks = []
        meta_list = []

        for i, (all_m, tgt_m) in enumerate(zip(all_humans_masks, target_masks)):
            rem_m, meta = self.compute_removal_mask(all_m, tgt_m)
            removal_masks.append(rem_m)
            meta_list.append(meta)

        return removal_masks, meta_list
