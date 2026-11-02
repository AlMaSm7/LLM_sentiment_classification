#!/bin/bash
#SBATCH --job-name=xed_data
#SBATCH --partition=regularmedium
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --output=outputs/jobs/data_%j.out
#SBATCH --error=outputs/jobs/data_%j.err

set -euo pipefail

# load the shared python environment
module load Python/3.12.3-GCCcore-13.3.0

# use scratch for python dependency caching
export UV_CACHE_DIR=/scratch/$USER/llm/uv_cache
export TMPDIR=/scratch/$USER/llm/tmp
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"

# use scratch for hugging face models
export HF_HOME=/scratch/$USER/llm/hf_cache

# start from the repository root
cd "$SLURM_SUBMIT_DIR"

# run source extraction and stratified splitting
uv run --locked --no-dev python scripts/run_data_preparation.py
