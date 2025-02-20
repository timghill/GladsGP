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

# Fig. 3: Assess PC truncation error
python -u plot_PC_RMSE.py ../train_config.py --nsim 16 32 64 128 256 512

# Fig. B3: Compare Original GlaDS simulations, PC low-rank representation and GP predictions
python -u plot_PC_maps.py ../train_config.py ../test_config.py 8 256

# Fig. 6: Plot error in more detail for a few configurations
python -u plot_integrated_RMSE.py ../train_config.py ../test_config.py

# Fig. 7--9: Width-averaged, scatter, timeseries test error
python -u plot_test_error.py ../test_config.py

# Table 3: Write txt table with performance statistics
python -u tabulate_performance.py ../test_config.py
