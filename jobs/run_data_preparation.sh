#!/bin/bash
#SBATCH --job-name=xed_data
#SBATCH --partition=regularmedium
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --output=outputs/jobs/data_%j.out
#SBATCH --error=outputs/jobs/data_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load Python/3.12.3-GCCcore-13.3.0

# set cache and temporary directories on scratch
export UV_CACHE_DIR="/scratch/$USER/llm/uv_cache"
export TMPDIR="/scratch/$USER/llm/tmp"
export HF_HOME="/scratch/$USER/llm/hf_cache"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR" "$HF_HOME"

# forward --stage and its value to Python
uv run --locked --no-dev --no-sync python scripts/run_data_preparation.py "$@"
