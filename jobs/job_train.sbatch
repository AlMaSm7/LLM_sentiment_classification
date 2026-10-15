#!/bin/bash

#SBATCH --job-name=xed_peft
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=02:00:00
#SBATCH --array=0-5%2
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

set -euo pipefail

# load python
module load Python/3.12.3-GCCcore-13.3.0

# use cached models on scratch - change to your username!!
export HF_HOME=/scratch/s6865704/llm/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# move to project directory
cd "$SLURM_SUBMIT_DIR"

# select experiment
LANGUAGES=(de de de sv sv sv)
CONDITIONS=(en target bilingual en target bilingual)
ID=$SLURM_ARRAY_TASK_ID

# train model
echo "training ${LANGUAGES[$ID]} - ${CONDITIONS[$ID]}"
uv run --locked python scripts/run_training.py "${LANGUAGES[$ID]}" "${CONDITIONS[$ID]}"
