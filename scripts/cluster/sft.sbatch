#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --error=runs/sft-%j.err
#SBATCH --output=runs/sft-%j.log
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
# V100 32GB (type_a/b/c), A100 80GB (type_e), H100 (type_h — update when confirmed)
#SBATCH --constraint="[type_a|type_b|type_c]"

# ── Environment ──
module load Python/Anaconda_v03.2023
source activate diffusiontuner

cd "$SLURM_SUBMIT_DIR"
mkdir -p runs

# ── Launch ──
# SLURM_GPUS gives us the number of GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch \
    --config_file configs/accelerate/multi_gpu.yaml \
    --num_processes "$SLURM_GPUS" \
    -m src.training.sft_trainer \
    --config configs/sft.json
