#!/bin/bash

#SBATCH --job-name=xed_eval
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --time=02:00:00
#SBATCH --array=0-41%2
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

set -euo pipefail

# load python
module load Python/3.12.3-GCCcore-13.3.0

# use cached models on scratch
export HF_HOME=/scratch/s6865704/llm/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# move to project directory
cd "$SLURM_SUBMIT_DIR"

# select evaluation split
SPLIT="${SPLIT:-dev}"
ID=$SLURM_ARRAY_TASK_ID

if [[ "$SPLIT" == dev ]]; then
    PER_LANG=21
    PROMPTS=16
elif [[ "$SPLIT" == test ]]; then
    PER_LANG=13
    PROMPTS=8
else
    echo "invalid split: $SPLIT" >&2
    exit 1
fi

# select language and experiment
LANGUAGES=(de sv)
LANG="${LANGUAGES[$((ID / PER_LANG))]}"
INDEX=$((ID % PER_LANG))

if (( ID < 0 || ID >= 2 * PER_LANG )); then
    echo "invalid array task: $ID" >&2
    exit 1
fi

if (( INDEX < PROMPTS )); then
    # evaluate prompting
    MODELS=(qwen phi)
    INPUTS=(en target)
    SHOTS=(0 5)

    if [[ "$SPLIT" == dev ]]; then
        VERSION=$((INDEX % 2 + 1))
        CASE=$((INDEX / 2))
    else
        VERSION=auto
        CASE=$INDEX
    fi

    MODEL="${MODELS[$((CASE / 4))]}"
    INPUT="${INPUTS[$((CASE % 2))]}"
    SHOT="${SHOTS[$(((CASE / 2) % 2))]}"

    uv run --locked python evaluate.py prompt "$LANG" "$INPUT" \
        --model "$MODEL" --shots "$SHOT" \
        --split "$SPLIT" --version "$VERSION"

else
    # evaluate peft
    N=$((INDEX - PROMPTS))
    CONDITIONS=(en en target bilingual bilingual)
    INPUTS=(en target target en target)

    uv run --locked python scripts/run_evaluation.py peft "$LANG" "${INPUTS[$N]}" \
        --condition "${CONDITIONS[$N]}" --split "$SPLIT"
fi
