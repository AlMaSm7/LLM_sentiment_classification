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
cd /scratch/s6865704/llm/LLM_sentiment_classification

# uncomment when model is downloaded / comment when model is not downloaded!!
# export HF_HUB_OFFLINE=1

export HF_HOME=/scratch/s6865704/llm/.cache/huggingface
mkdir -p "$HF_HOME"


if [[ -f .env ]]; then
    set -a
    source .env
    set +a
    echo "Loaded environment variables from .env"
else
    echo "Warning: .env not found; continuing without local credentials."
fi


# Load CUDA for PyTorch / GPU training
module purge
module load CUDA/12.1.1

# Show assigned GPU
nvidia-smi

source .venv/bin/activate

python --version

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


uv run python -u scripts/run_peft.py \
    --language "$LANGUAGE" \
    --condition "$CONDITION"