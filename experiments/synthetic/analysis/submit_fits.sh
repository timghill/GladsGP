#!/bin/bash
#SBATCH --job-name="fit"
#SBATCH --time=0-03:00
#SBATCH --mem=64G
#SBATCH --account=def-gflowers
#SBATCH --mail-user=tha111@sfu.ca
#SBATCH --mail-type=FAIL,END

source ~/SFU-code/GladsGP/pyenv/bin/activate

python fit_all_models.py IS/config.py
