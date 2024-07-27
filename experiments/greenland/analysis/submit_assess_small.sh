#!/bin/bash
#SBATCH --job-name="assess-small"
#SBATCH --time=0-16:00
#SBATCH --mem=48G
#SBATCH --account=def-gflowers
#SBATCH --mail-user=tha111@sfu.ca
#SBATCH --mail-type=FAIL,END
#SBATCH --output=assess-small.out
#SBATCH --error=assess-small.err

source ../issm/setenv.sh
source ~/SFU-code/GladsGP/pyenv/bin/activate

python -u assess_all_models.py ../train_config.py ../test_config.py --npc {1..11} --nsim 16 32 64 128 -r
