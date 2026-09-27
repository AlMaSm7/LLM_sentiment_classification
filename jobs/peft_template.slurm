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

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
    echo "Loaded environment variables from .env"
else
    echo "Warning: .env not found; continuing without local credentials."
fi

echo "Starting training job at $(date): PEFT experiments for English and German for LoRa and QLoRA"

RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

uv run python scripts/train_peft.py \
    --config configs/peft_template.yaml \
    --language de \
    --condition english \
    --run-name "lora_de_english_${RUN_TIMESTAMP}"

# German-only PEFT.
uv run python scripts/train_peft.py \
    --config configs/peft_template.yaml \
    --language de \
    --condition target \
    --run-name "lora_de_target_${RUN_TIMESTAMP}"

# Bilingual PEFT using the matched English-German training pairs.
uv run python scripts/train_peft.py \
    --config configs/peft_template.yaml \
    --language de \
    --condition bilingual \
    --run-name "lora_de_bilingual_${RUN_TIMESTAMP}"

echo "Completed PEFT experiments for loRA at $(date)"


uv run python scripts/train_peft.py \
    --config configs/peft_qlora.yaml \
    --language de \
    --condition english \
    --run-name "qlora_de_english_${RUN_TIMESTAMP}"

# German-only PEFT.
uv run python scripts/train_peft.py \
    --config configs/peft_qlora.yaml \
    --language de \
    --condition target \
    --run-name "qlora_de_target_${RUN_TIMESTAMP}"

# Bilingual PEFT using the matched English-German training pairs.
uv run python scripts/train_peft.py \
    --config configs/peft_qlora.yaml \
    --language de \
    --condition bilingual \
    --run-name "qlora_de_bilingual_${RUN_TIMESTAMP}"

echo "Completed PEFT experiments for loRA at $(date)"


