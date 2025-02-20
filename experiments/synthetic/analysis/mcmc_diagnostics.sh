#!/bin/bash
#SBATCH --job-name="mcmc-diagnostics"
#SBATCH --time=00-00:30
#SBATCH --mem=8G
#SBATCH --account=def-gflowers
#SBATCH --mail-user=tha111@sfu.ca
#SBATCH --mail-type=FAIL,END,BEGIN
#SBATCH --output=mcmc_diagnostics.out
#SBATCH --error=mcmc_diagnostics.err

source ~/SFU-code/GladsGP/pyenv/bin/activate

python -u mcmc_diagnostics.py ../train_config.py --recompute