#!/bin/bash

source ~/SFU-code/GladsGP/pyenv/bin/activate

cd ../
python -u assess_all_models.py ../train_config.py ../test_config.py --npc {1..11} --nsim 16 32 64 128 256 512

cd test/
