#!/bin/bash

#SBATCH --job-name=prompting
#SBATCH --array=0-1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --output=outputs/jobs/peft_%A_%a.out
#SBATCH --error=outputs/jobs/peft_%A_%a.err


cd /scratch/s6866425/llm_project

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
    echo "Loaded environment variables from .env"
else
    echo "Warning: .env not found; continuing without local credentials."
fi

module load CUDA/12.1.1
source .venv/bin/activate

# Permutations of Language, Target, and Mode
EXPERIMENTS=(
    "de english zero_shot"
    "de english five_shot"
    "de target zero_shot"
    "de target five_shot"
)

# Parse the three variables for the current array task
EXPERIMENT=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}
read LANGUAGE TARGET MODE <<< "$EXPERIMENT"

echo "Running Prompting Evaluation:"
echo "Language: $LANGUAGE"
echo "Target Dataset: $TARGET"
echo "Mode: $MODE"

# Execute the python script with the new mode argument
uv run python scripts/run_prompting.py \
    --language "$LANGUAGE" \
    --target "$TARGET" \
    --split test \
    --mode "$MODE"