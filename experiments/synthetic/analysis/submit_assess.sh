#!/bin/bash
#SBATCH --job-name="assess"
#SBATCH --time=0-8:00
#SBATCH --mem=48G
#SBATCH --account=def-gflowers
#SBATCH --mail-user=tha111@sfu.ca
#SBATCH --mail-type=FAIL,END
#SBATCH --output=assess.out
#SBATCH --error=assess.err

source ../issm/setenv.sh
source ~/SFU-code/GladsGP/pyenv/bin/activate

python -u assess_all_models.py ../train_config.py ../test_config.py --npc {1..11} --nsim 512 256 128 64 32 16 -r
