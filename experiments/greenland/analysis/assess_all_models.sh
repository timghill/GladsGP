#!/bin/bash
#SBATCH --account=def-gflowers
#SBATCH --time=0-00:30
#SBATCH --mem=4G

source ~/SFU-code/GladsGP/pyenv/bin/activate

python -u assess_all_models.py ../train_config.py ../test_config.py --npc {1..11} --nsim 16 32 64 128 256 512
