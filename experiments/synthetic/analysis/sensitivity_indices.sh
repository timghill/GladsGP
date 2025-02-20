#!/bin/bash
#SBATCH --job-name="sensitivity"
#SBATCH --time=0-02:00
#SBATCH --mem=10G
#SBATCH --account=def-gflowers
#SBATCH --mail-user=tha111@sfu.ca
#SBATCH --mail-type=FAIL,END
#SBATCH --output=sensitivity_indices.out
#SBATCH --error=sensitivity_indices.err

source ~/SFU-code/GladsGP/pyenv/bin/activate

python -u sensitivity_indices.py ../train_config.py -r
