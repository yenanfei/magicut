"""
DiffuEraser Multi-Model Weight Downloader
Downloads all required deep learning models:
1. DiffuEraser (BrushNet + Motion Adapter weights) - COMPLETED
2. Stable Diffusion v1-5 components
3. SD VAE (sd-vae-ft-mse)
4. PCM_Weights (2-Step / 4-Step fast diffusion)
5. ProPainter priori models
"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def download_weights(target_dir: str = "weights"):
    os.makedirs(target_dir, exist_ok=True)
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print("=" * 60)
    print("[DiffuEraser] Starting Multi-Model Fusion Weight Download")
    print(f"Target Directory: {os.path.abspath(target_dir)}")
    print("=" * 60)

    from huggingface_hub import snapshot_download as hf_snapshot_download
    from modelscope import snapshot_download as ms_snapshot_download

    # 1. DiffuEraser Main Weights (Check if exists)
    diffueraser_dir = os.path.join(target_dir, 'diffuEraser')
    if not os.path.exists(diffueraser_dir):
        print("\n[1/5] Downloading DiffuEraser weights (from ModelScope)...")
        ms_snapshot_download('xingzi/diffuEraser', local_dir=diffueraser_dir)
    else:
        print("\n[1/5] DiffuEraser weights already exist, skipping.")

    # 2. SD VAE
    print("\n[2/5] Downloading SD-VAE (sd-vae-ft-mse)...")
    try:
        hf_snapshot_download(repo_id="stabilityai/sd-vae-ft-mse", local_dir=os.path.join(target_dir, 'sd-vae-ft-mse'))
    except Exception as e:
        print(f"Fallback VAE download via ModelScope: {e}")
        ms_snapshot_download('stabilityai/sd-vae-ft-mse', local_dir=os.path.join(target_dir, 'sd-vae-ft-mse'))

    # 3. PCM Fast Diffusion Weights
    print("\n[3/5] Downloading PCM-Weights...")
    hf_snapshot_download(repo_id="wangfuyun/PCM_Weights", local_dir=os.path.join(target_dir, 'PCM_Weights'))

    # 4. ProPainter Priori Weights
    print("\n[4/5] Downloading ProPainter priori weights...")
    ms_snapshot_download('Fluchw/propainter', local_dir=os.path.join(target_dir, 'propainter'))

    # 5. SD 1.5 Base Components
    print("\n[5/5] Downloading Stable-Diffusion-v1-5 components...")
    hf_snapshot_download(
        repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        local_dir=os.path.join(target_dir, 'stable-diffusion-v1-5'),
        allow_patterns=['feature_extractor/*', 'model_index.json', 'safety_checker/*', 'scheduler/*', 'text_encoder/*', 'tokenizer/*', 'unet/*']
    )

    print("\n[SUCCESS] All DiffuEraser model weights downloaded successfully!")

if __name__ == "__main__":
    download_weights()
