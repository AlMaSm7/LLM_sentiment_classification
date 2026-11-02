#!/bin/bash
#SBATCH --job-name=results
#SBATCH --partition=regularshort
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/jobs/%x_%j.out
#SBATCH --error=outputs/jobs/%x_%j.err

set -euo pipefail

# load python
module load Python/3.12.3-GCCcore-13.3.0

# run from the project root
cd "$SLURM_SUBMIT_DIR"

# keep uv temporary files out of the home directory
export UV_CACHE_DIR="/scratch/$USER/llm/uv_cache"
export TMPDIR="/scratch/$USER/llm/tmp"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"

# use scratch for hugging face models
export HF_HOME=/scratch/$USER/llm/hf_cache

# render matplotlib figures without a display
export MPLBACKEND=Agg

# forward arguments to the results script
uv run --locked --no-dev --no-sync python scripts/run_results.py "$@"
