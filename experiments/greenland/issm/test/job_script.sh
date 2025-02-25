#!/bin/bash
#SBATCH --job-name="Gr-test"
#SBATCH --time=0-24:00
#SBATCH --mem=2G
#SBATCH --account=def-gflowers
#SBATCH --mail-user=tha111@sfu.ca
#SBATCH --mail-type=FAIL,END

source ../setenv.sh
task.run
