#!/bin/bash

#SBATCH --job-name=prompting
#SBATCH --array=0-7
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

EXPERIMENTS=(
    "de primary zero_shot"
    "de primary five_shot"
    "de secondary zero_shot"
    "de secondary five_shot"
    "sv primary zero_shot"
    "sv primary five_shot"
    "sv secondary zero_shot"
    "sv secondary five_shot"
)

EXPERIMENT="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"

read LANGUAGE MODEL METHOD <<< "$EXPERIMENT"


echo "Language: $LANGUAGE"
echo "Model:    $MODEL"
echo "Method:   $METHOD"


python -u scripts/run_prompting.py \
    --language "$LANGUAGE" \
    --model "$MODEL" \
    --method "$METHOD"