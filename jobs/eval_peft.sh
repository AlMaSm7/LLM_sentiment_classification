#!/bin/bash
# Evaluate trained PEFT adapters on dev or test, without retraining.
# Submit from the project root:
#     sbatch --array=0,1 jobs/eval_peft.sh dev
#     sbatch --array=0,1 jobs/eval_peft.sh test      # ONCE, after the test freeze
# Array index -> experiment, in the same order as EXPERIMENTS below.
#SBATCH --job-name=eval_peft
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --output=outputs/jobs/eval_peft_%A_%a.out
#SBATCH --error=outputs/jobs/eval_peft_%A_%a.err

cd "$SLURM_SUBMIT_DIR"

module load CUDA/12.1.1
source /scratch/$USER/.llm/bin/activate
export HF_HOME=/scratch/$USER/hf_cache
export WANDB_MODE=disabled

nvidia-smi --query-gpu=name --format=csv,noheader

SPLIT="${1:?usage: sbatch --array=... jobs/eval_peft.sh dev|test}"
shift   # any further arguments (e.g. --overwrite) are passed on

EXPERIMENTS=(
    "sv english"
    "sv target"
    "sv bilingual"
)
read -r LANGUAGE CONDITION <<< "${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"
echo "Language: $LANGUAGE  Condition: $CONDITION  Split: $SPLIT"

python scripts/evaluate_peft.py --language "$LANGUAGE" --condition "$CONDITION" --split "$SPLIT" "$@"