#!/bin/bash
#SBATCH --job-name="train"
#SBATCH --time=0-24:00
#SBATCH --mem=2G
#SBATCH --account=def-gflowers
#SBATCH --mail-user=tha111@sfu.ca
#SBATCH --mail-type=FAIL,END

# Time calculation
# Each job: ~1.5 hours for 2 years of simulation time

# Don't change this line:

source ../setenv.sh
source $PYENV/bin/activate
task.run
