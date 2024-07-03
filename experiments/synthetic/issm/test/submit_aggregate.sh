#!/bin/bash
#SBATCH --time=0-00:30
#SBATCH --account=def-gflowers
#SBATCH --mem=2G
#SBATCH --ntasks=1
#SBATCH --output=aggregate.out
#SBATCH --error=aggregate.err

source ../setenv.sh

python -u -m src.aggregate_outputs issm ../../test_config.py 100

