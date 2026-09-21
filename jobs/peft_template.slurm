#!/bin/bash

#SBATCH --job-name=xed-peft
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-%j.out

set -euo pipefail

# 1. Load system modules if your environment relies on them
module load CUDA/12.1.1

# 2. Navigate to your Scratch directory, NOT Home
cd /scratch/s6866425/llm_project

export WANDB_API_KEY="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"
export HF_TOKEN="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"

uv run python scripts/train_peft.py \
    --config configs/peft_template.yaml \
    --language de \
    --condition english