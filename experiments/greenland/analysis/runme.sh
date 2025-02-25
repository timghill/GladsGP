#!/bin/bash
#SBATCH --job-name=post-proc
#SBATCH --account=def-gflowers
#SBATCH --time=0-00:10
#SBATCH --mem=8G
#SBATCH --output=runme.out
#SBATCH --error=runme.err

# Run all analysis for synthetic ice-sheet GP emulation experiment
# REQUIRES fit/ and test/ directories to have been run

source ~/SFU-code/GladsGP/pyenv/bin/activate

set -x

python -u plot_PC_basis.py ../train_config.py

python -u plot_test_error.py ../train_config.py ../test_config.py

python -u tabulate_performance.py ../train_config.py ../test_config.py --nsim 256 --npc 9
