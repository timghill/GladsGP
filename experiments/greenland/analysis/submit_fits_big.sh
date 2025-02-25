#!/bin/bash
#SBATCH --job-name="fits-big"
#SBATCH --time=0-06:00
#SBATCH --mem=48G
#SBATCH --account=def-gflowers
#SBATCH --mail-user=tha111@sfu.ca
#SBATCH --mail-type=FAIL,END

source ../issm/setenv.sh
source ~/SFU-code/GladsGP/pyenv/bin/activate

python -u fit_all_models.py ../train_config.py --npc {8..11} --nsim 256 -r
