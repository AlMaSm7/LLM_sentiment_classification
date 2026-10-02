#!/bin/bash
# Phase 2: zero-shot and five-shot prompting.
# Submit from the project root:   sbatch jobs/run_prompting.sh dev
#                                  sbatch jobs/run_prompting.sh dev --sensitivity
#                                  sbatch jobs/run_prompting.sh test
#SBATCH --job-name=prompting
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=jobs/logs/%x_%j.out

set -euo pipefail

export HF_HOME=/scratch/$USER/hf_cache      # model cache + login token
export UV_CACHE_DIR=/scratch/$USER/uv_cache
PYTHON=/scratch/$USER/.llm/bin/python       # existing environment, no uv sync needed

SPLIT="${1:-dev}"
shift || true          # further arguments (e.g. --sensitivity) are passed on

"$PYTHON" scripts/run_prompting.py --split "$SPLIT" "$@"
