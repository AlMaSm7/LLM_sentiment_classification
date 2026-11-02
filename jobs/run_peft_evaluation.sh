#!/bin/bash
# evaluate trained PEFT adapters on dev or test, without retraining
# submit from the project root:
#     sbatch --array=0,1 jobs/eval_peft.sh dev
#     sbatch --array=0,1 jobs/eval_peft.sh test      # ONCE, after the test freeze
# array index -> experiment, in the same order as EXPERIMENTS below
#SBATCH --job-name=eval_peft
#SBATCH --array=0-5%2
#SBATCH --time=02:00:00
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=outputs/jobs/%x_%j.out
#SBATCH --error=outputs/jobs/%x_%j.err

cd "$SLURM_SUBMIT_DIR"

module load Python/3.12.3-GCCcore-13.3.0

# start from the repository root
cd "$SLURM_SUBMIT_DIR"

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
    echo "Loaded environment variables from .env"
else
    echo "Warning: .env not found; continuing without local credentials."
fi

# use scratch for python dependency caching
export UV_CACHE_DIR=/scratch/$USER/llm/uv_cache
export TMPDIR=/scratch/$USER/llm/tmp
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"

# use scratch for hugging face models
export HF_HOME=/scratch/$USER/llm/hf_cache

export WANDB_MODE=disabled

nvidia-smi --query-gpu=name --format=csv,noheader

SPLIT="${1:?usage: sbatch --array=... jobs/eval_peft.sh dev|test}"
shift   # any further arguments (e.g. --overwrite) are passed on

EXPERIMENTS=(
    "de english"
    "de target"
    "de bilingual"
    "sv english"
    "sv target"
    "sv bilingual"
)
read -r LANGUAGE CONDITION <<< "${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"
echo "Language: $LANGUAGE  Condition: $CONDITION  Split: $SPLIT"

uv run --locked --no-dev python scripts/run_peft_evaluation.py --language "$LANGUAGE" --condition "$CONDITION" --split "$SPLIT" "$@"