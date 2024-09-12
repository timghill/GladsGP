#!/bin/bash

# Test the python environment installation by running analysis scripts

set -x

cd experiments/synthetic/analysis

python assess_all_models.py ../train_config.py ../test_config.py --npc {1..11} --nsim 16 32 64 128 256 512
python plot_test_error.py ../train_config.py ../test_config.py
