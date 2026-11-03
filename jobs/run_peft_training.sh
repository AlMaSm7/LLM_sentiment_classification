#!/bin/bash

#SBATCH --job-name=peft
#SBATCH --array=0-2%2
#SBATCH --time=02:00:00
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=outputs/jobs/%x_%j.out
#SBATCH --error=outputs/jobs/%x_%j.err


set -euo pipefail

# start from the project root
cd "$SLURM_SUBMIT_DIR"

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
    echo "Loaded environment variables from .env"
else
    echo "Warning: .env not found; continuing without local credentials."
fi

# show assigned GPU
nvidia-smi

# load CUDA for PyTorch / GPU training
module load Python/3.12.3-GCCcore-13.3.0

# use scratch for python dependency caching
export UV_CACHE_DIR=/scratch/$USER/llm/uv_cache
export TMPDIR=/scratch/$USER/llm/tmp
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"

# use scratch for hugging face models
export HF_HOME=/scratch/$USER/llm/hf_cache

# all experiments
EXPERIMENTS=(
    # "de english"
    # "de target"
    # "de bilingual"
    "sv english"
    "sv target"
    "sv bilingual"
)

# pick the experiment belonging to this array task
EXPERIMENT=${EXPERIMENTS[${SLURM_ARRAY_TASK_ID:?submit using sbatch --array}]}


# split it into language and condition
read LANGUAGE CONDITION <<< "$EXPERIMENT"


echo "Running experiment:"
echo "Language:  $LANGUAGE"
echo "Condition: $CONDITION"


uv run --locked --no-dev python scripts/run_peft_training.py \
    --language "$LANGUAGE" \
    --condition "$CONDITION"