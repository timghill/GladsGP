#!/bin/bash
#SBATCH --job-name="assess-big"
#SBATCH --time=0-16:00
#SBATCH --mem=48G
#SBATCH --account=def-gflowers
#SBATCH --mail-user=tha111@sfu.ca
#SBATCH --mail-type=FAIL,END
#SBATCH --output=assess-big.out
#SBATCH --error=assess-big.err

source ../issm/setenv.sh
source ~/SFU-code/GladsGP/pyenv/bin/activate

python -u assess_all_models.py ../train_config.py ../test_config.py --npc {1..11} --nsim 512 256 -r
