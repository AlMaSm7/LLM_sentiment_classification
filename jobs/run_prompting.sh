#!/bin/bash
# phase 2: zero-shot and five-shot prompting
# submit from the project root:   sbatch jobs/run_prompting.sh dev
#                                  sbatch jobs/run_prompting.sh dev --sensitivity
#                                  sbatch jobs/run_prompting.sh test
#SBATCH --job-name=prompting
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/jobs/%x_%j.out
#SBATCH --error=outputs/jobs/%x_%j.err

set -euo pipefail

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

SPLIT="${1:-dev}"
shift || true          # further arguments (e.g. --sensitivity) are passed on

cd "$SLURM_SUBMIT_DIR"
mkdir -p jobs/logs
uv run --locked python scripts/run_prompting.py --split "$SPLIT" "$@"
