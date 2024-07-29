#!/bin/bash
#SBATCH --job-name="test-error"
#SBATCH --time=0-4:00
#SBATCH --mem=32G
#SBATCH --account=def-gflowers
#SBATCH --mail-user=tha111@sfu.ca
#SBATCH --mail-type=FAIL,END
#SBATCH --output=test_err.out
#SBATCH --error=test_err.err

source ../issm/setenv.sh
source ~/SFU-code/GladsGP/pyenv/bin/activate

python plot_test_error.py ../train_config.py ../test_config.py
