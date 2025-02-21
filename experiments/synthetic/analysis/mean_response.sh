#!/bin/bash
#SBATCH --job-name="mean-response"
#SBATCH --time=00-01:00
#SBATCH --mem=2G
#SBATCH --account=def-gflowers
#SBATCH --mail-user=tha111@sfu.ca
#SBATCH --mail-type=FAIL,END
#SBATCH --output=mean_response.out
#SBATCH --error=mean_response.err

source ~/SFU-code/GladsGP/pyenv/bin/activate

python -u mean_response.py ../train_config.py --recompute
