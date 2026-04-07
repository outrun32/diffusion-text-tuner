#!/bin/bash
# Run this on a node WITH internet access to set up the conda env + cache models.
# The cluster compute nodes have NO internet.
#
# Usage (from login node or local machine):
#   bash scripts/cluster/setup_env.sh

set -euo pipefail

echo "=== Setting up diffusiontuner environment ==="

module load Python/Anaconda_v03.2023

# Create conda env
conda create -n diffusiontuner python=3.11 -y || true
source activate diffusiontuner

# Install PyTorch + deps
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install accelerate transformers diffusers peft bitsandbytes
pip install tqdm pillow

echo ""
echo "=== Pre-caching models (REQUIRED: no internet on compute nodes) ==="
echo "Run these Python commands to download models to HuggingFace cache:"
echo ""
echo "  python -c \"from diffusers import Flux2KleinPipeline; Flux2KleinPipeline.from_pretrained('black-forest-labs/FLUX.2-klein-base-4B')\""
echo "  python -c \"from transformers import AutoModelForImageTextToText, AutoProcessor; AutoProcessor.from_pretrained('Qwen/Qwen3.5-9B'); AutoModelForImageTextToText.from_pretrained('Qwen/Qwen3.5-9B')\""
echo ""
echo "Environment ready. Submit jobs with: sbatch scripts/cluster/sft.sbatch"
