"""
MagiCut DiffuEraser Diffusion Model Inpainting Adapter
Integrates Alibaba Tongyi Lab's DiffuEraser (Diffusion + BrushNet + ProPainter Prior)
for photorealistic, flicker-free video object removal and generative background reconstruction.
"""

import os
import sys
import cv2
import numpy as np
import torch
from typing import List, Optional, Tuple


class DiffuEraserAdapter:
    def __init__(
        self,
        diffueraser_root: str = "third_party/DiffuEraser",
        weights_dir: str = "weights",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        pcm_steps: str = "2-Step"
    ):
        self.diffueraser_root = os.path.abspath(diffueraser_root)
        self.weights_dir = os.path.abspath(weights_dir)
        self.device = device
        self.pcm_steps = pcm_steps
        self.model = None
        self.is_ready = False

        # Add diffueraser root to python search path
        if self.diffueraser_root not in sys.path:
            sys.path.insert(0, self.diffueraser_root)

        self._check_weights()

    def _check_weights(self):
        """
        Checks if required diffusion model weights exist.
        """
        required_dirs = [
            os.path.join(self.weights_dir, "diffuEraser"),
            os.path.join(self.weights_dir, "stable-diffusion-v1-5"),
            os.path.join(self.weights_dir, "sd-vae-ft-mse"),
            os.path.join(self.weights_dir, "propainter")
        ]
        all_exist = all(os.path.exists(d) for d in required_dirs)
        if all_exist:
            self.is_ready = True
            print("[DiffuEraser] All diffusion model weights verified.")
        else:
            print("[DiffuEraser] Model weights need to be downloaded to weights/ folder before running diffusion inference.")

    def run_diffueraser_pipeline(
        self,
        input_video_path: str,
        removal_mask_video_path: str,
        output_video_path: str,
        max_frames: int = 150,
        max_img_size: int = 720,
        progress_cb=None
    ) -> str:
        """
        Runs the full DiffuEraser multi-model fusion pipeline:
        1. Generates ProPainter prior video
        2. Denoises via DiffuEraser UNet + BrushNet branch + PCM 2-Step sampler
        """
        if progress_cb:
            progress_cb(0.1, "Initializing DiffuEraser Diffusion Pipeline...")

        os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
        results_dir = os.path.dirname(os.path.abspath(output_video_path))
        priori_path = os.path.join(results_dir, "priori_temp.mp4")

        try:
            from diffueraser.diffueraser import DiffuEraser
            from propainter.inference import Propainter, get_device

            dev = get_device()
            base_model = os.path.join(self.weights_dir, "stable-diffusion-v1-5")
            vae_path = os.path.join(self.weights_dir, "sd-vae-ft-mse")
            diffueraser_path = os.path.join(self.weights_dir, "diffuEraser")
            propainter_dir = os.path.join(self.weights_dir, "propainter")

            if progress_cb:
                progress_cb(0.25, "Running Step 1: ProPainter Temporal Motion Prior...")

            # 1. Priori Model (ProPainter)
            propainter = Propainter(propainter_dir, device=dev)
            propainter.forward(
                input_video_path,
                removal_mask_video_path,
                priori_path,
                video_length=max_frames,
                ref_stride=10,
                neighbor_length=10,
                subvideo_length=30,
                mask_dilation=6
            )
            del propainter
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            if progress_cb:
                progress_cb(0.60, "Running Step 2: DiffuEraser Video Diffusion + BrushNet Generative Synthesis...")

            # 2. DiffuEraser Diffusion
            video_inpainting_sd = DiffuEraser(
                dev,
                base_model,
                vae_path,
                diffueraser_path,
                ckpt=self.pcm_steps
            )

            video_inpainting_sd.forward(
                input_video_path,
                removal_mask_video_path,
                priori_path,
                output_video_path,
                max_img_size=min(max_img_size, 640),
                video_length=max_frames,
                mask_dilation_iter=6,
                nframes=8,
                guidance_scale=0.0
            )

            del video_inpainting_sd
            gc.collect()
            torch.cuda.empty_cache()

            if os.path.exists(priori_path):
                os.remove(priori_path)

            if progress_cb:
                progress_cb(1.0, "DiffuEraser Generative Inpainting Complete!")

            return output_video_path

        except Exception as e:
            print(f"[DiffuEraser] Error during diffusion inference: {e}")
            raise e
