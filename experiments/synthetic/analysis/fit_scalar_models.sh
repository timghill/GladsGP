#!/bin/bash
#SBATCH --job-name=scalar
#SBATCH --account=def-gflowers
#SBATCH --time=0-12:00
#SBATCH --mem=8G
#SBATCH --mail-user=tha111@sfu.ca
#SBATCH --mail-type=FAIL,END
#SBATCH --output=fit_scalar_models.out
#SBATCH --error=fit_scalar_models.err


source ~/SFU-code/GladsGP/pyenv/bin/activate

python -u fit_scalar_models.py --nsim 16 32 64 128 256 512 --recompute ../train_config.py ../test_config.py
