#!/bin/bash
#SBATCH --job-name="assess"
#SBATCH --time=0-12:00
#SBATCH --mem=36G
#SBATCH --account=def-gflowers
#SBATCH --mail-user=tha111@sfu.ca
#SBATCH --mail-type=FAIL,END
#SBATCH --output=assess.out
#SBATCH --error=assess.err

soure ../issm/setenv.sh
source ~/SFU-code/GladsGP/pyenv/bin/activate

python -u assess_all_models.py ../train_config.py ../test_config.py --npc {1..11} --nsim 16 32 64 128 256 -r
