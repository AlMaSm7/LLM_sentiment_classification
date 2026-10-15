#!/bin/bash

#SBATCH --job-name=xed_data
#SBATCH --output=logs/data_%j.out
#SBATCH --error=logs/data_%j.err
#SBATCH --partition=regularshort
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

# load python
module load Python/3.12.3-GCCcore-13.3.0

# move to project directory
cd "$SLURM_SUBMIT_DIR"

# download german and swedish data
LANGUAGES="de sv" bash scripts/get_data.sh
