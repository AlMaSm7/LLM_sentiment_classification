#!/bin/bash

#SBATCH --job-name=peft
#SBATCH --array=0-5
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --output=outputs/jobs/peft_%A_%a.out
#SBATCH --error=outputs/jobs/peft_%A_%a.err


# CHANGE TO YOUR USER
cd /scratch/s6866425/llm_project


if [[ -f .env ]]; then
    set -a
    source .env
    set +a
    echo "Loaded environment variables from .env"
else
    echo "Warning: .env not found; continuing without local credentials."
fi

# Show assigned GPU
nvidia-smi

# Load CUDA for PyTorch / GPU training
module load CUDA/12.1.1

source .venv/bin/activate

# All six experiments.
EXPERIMENTS=(
    "de english"
    "de target"
    "de bilingual"
    "sv english"
    "sv target"
    "sv bilingual"
)

# Pick the experiment belonging to this array task.
EXPERIMENT=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}


# Split it into language and condition.
read LANGUAGE CONDITION <<< "$EXPERIMENT"


echo "Running experiment:"
echo "Language:  $LANGUAGE"
echo "Condition: $CONDITION"


uv run python scripts/run_peft.py \
    --language "$LANGUAGE" \
    --condition "$CONDITION"