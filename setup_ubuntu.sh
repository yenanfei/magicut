#!/usr/bin/env bash
# ==============================================================================
# MagiCut - Ubuntu Server One-Click Environment Setup Script
# Target: Ubuntu 20.04 / 22.04 / 24.04 with NVIDIA GPU (A10, A100, RTX 3090/4090)
# ==============================================================================

set -e

echo "=========================================================="
echo "🚀 [MagiCut] Starting Ubuntu GPU Environment Setup"
echo "=========================================================="

# 1. System dependencies
echo "\n[1/5] Installing system packages (ffmpeg, git, libgl1)..."
sudo apt-get update -y
sudo apt-get install -y ffmpeg libsm6 libxext6 libgl1-mesa-glx git curl build-essential

# 2. Python & PyTorch CUDA setup
echo "\n[2/5] Installing PyTorch with CUDA support..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Core dependencies
echo "\n[3/5] Installing MagiCut and DiffuEraser dependencies..."
pip install "transformers>=4.41.1,<4.45.0" "diffusers==0.29.2" "peft==0.13.2" \
    accelerate modelscope einops scipy av opencv-python ultralytics gradio pyyaml requests

# 4. Clone third_party DiffuEraser if not present
echo "\n[4/5] Checking DiffuEraser submodule..."
if [ ! -d "third_party/DiffuEraser/.git" ]; then
    mkdir -p third_party
    git clone https://github.com/alibaba/DiffuEraser.git third_party/DiffuEraser
fi

# 5. Link weights folder
mkdir -p weights
mkdir -p third_party/DiffuEraser/weights
rm -rf third_party/DiffuEraser/weights
ln -s "$(pwd)/weights" "$(pwd)/third_party/DiffuEraser/weights"

echo "\n[5/5] Downloading Multi-Model Fusion Weights..."
python download_diffueraser_weights.py

echo "\n=========================================================="
echo "🎉 [MagiCut] Ubuntu GPU Environment Ready for AGY Experiments!"
echo "=========================================================="
