#!/bin/bash
#SBATCH --job-name="sensitivity"
#SBATCH --time=0-12:00
#SBATCH --mem=16G
#SBATCH --account=def-gflowers
#SBATCH --mail-user=tha111@sfu.ca
#SBATCH --mail-type=FAIL,END
#SBATCH --output=sensitivity.out
#SBATCH --error=sensitivity.err

source ../issm/setenv.sh
source ~/SFU-code/GladsGP/pyenv/bin/activate

python -u sensitivity_indices.py ../train_config.py -r
