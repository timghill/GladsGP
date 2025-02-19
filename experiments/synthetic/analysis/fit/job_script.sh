#!/bin/bash
#SBATCH --job-name="fit-jobs"
#SBATCH --time=0-01:00
#SBATCH --mem=16G
#SBATCH --account=def-gflowers
#SBATCH --mail-user=tha111@sfu.ca
#SBATCH --mail-type=FAIL,END

# Don't change this line:

source ~/SFU-code/GladsGP/pyenv/bin/activate

task.run
