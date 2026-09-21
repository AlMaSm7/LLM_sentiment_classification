#!/bin/bash

# idk if this is correct, pls verify

#SBATCH --job-name=xed-peft
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd /path/to/LLM_sentiment_classification

uv run python scripts/train_peft.py \
    --config configs/peft_template.yaml \
    --language de \
    --condition english
