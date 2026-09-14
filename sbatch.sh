#!/bin/sh

# specify project-related resources
#SBATCH --job-name=name_of_project
#SBATCH --time=4:00:00
#SBATCH --mem=300000
#SBATCH --gpus-per-node=v100:1

# check the number of GPUs being used
nvidia-smi

# Load the relevant modules
# these depend on the modules used in your application!
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1

# activate the python environment
source /scratch/s6866425/test_project/.venv/bin/activate

# Define the Python script to be executed
PYTHON_SCRIPT="./test.py"

# Run the Python script
python3 "$PYTHON_SCRIPT"