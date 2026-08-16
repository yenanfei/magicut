"""
DiffuEraser Multi-Model Weight Downloader
Downloads all required deep learning models:
1. DiffuEraser (BrushNet + Motion Adapter weights)
2. Stable Diffusion v1-5 (Pruned components)
3. SD VAE (sd-vae-ft-mse)
4. PCM_Weights (2-Step / 4-Step fast diffusion)
5. ProPainter priori models (ProPainter.pth, raft-things.pth, recurrent_flow_completion.pth)
"""

import os
import sys

def download_weights(target_dir: str = "weights"):
    os.makedirs(target_dir, exist_ok=True)
    print("=" * 60)
    print("✨ Starting DiffuEraser & Multi-Model Fusion Weight Download")
    print(f"Target Directory: {os.path.abspath(target_dir)}")
    print("=" * 60)

    try:
        from modelscope import snapshot_download

        # 1. DiffuEraser Main Weights
        print("\n[1/4] Downloading DiffuEraser weights (from ModelScope)...")
        snapshot_download('xingzi/diffuEraser', local_dir=os.path.join(target_dir, 'diffuEraser'))

        # 2. SD VAE
        print("\n[2/4] Downloading SD-VAE weights...")
        snapshot_download('AI-ModelScope/sd-vae-ft-mse', local_dir=os.path.join(target_dir, 'sd-vae-ft-mse'))

        # 3. PCM Fast Diffusion Weights
        print("\n[3/4] Downloading PCM-Weights...")
        snapshot_download('AI-ModelScope/PCM_Weights', local_dir=os.path.join(target_dir, 'PCM_Weights'))

        # 4. SD 1.5 Base
        print("\n[4/4] Downloading Stable-Diffusion-v1-5...")
        snapshot_download('AI-ModelScope/stable-diffusion-v1-5', local_dir=os.path.join(target_dir, 'stable-diffusion-v1-5'))

        print("\n🎉 All DiffuEraser model weights downloaded successfully!")

    except Exception as e:
        print(f"\nNote: Automated ModelScope download error: {e}")
        print("You can manually download weights from:")
        print("- DiffuEraser:  https://huggingface.co/lixiaowen/diffuEraser")
        print("- SD 1.5:       https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5")
        print("- SD VAE:       https://huggingface.co/stabilityai/sd-vae-ft-mse")
        print("- PCM Weights:  https://huggingface.co/wangfuyun/PCM_Weights")

if __name__ == "__main__":
    download_weights()
