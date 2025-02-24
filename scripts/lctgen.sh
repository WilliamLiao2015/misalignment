#!/bin/bash

# Activate Miniconda Environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lctgen

# Print the Current Working Directory
echo "Running LCTGen"
echo "Current directory: $(pwd)"

# Execute the Command
export PYTHONPATH=$(pwd)
python -m models.lctgen
