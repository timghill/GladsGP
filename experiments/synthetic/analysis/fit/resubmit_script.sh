#!/bin/bash
#SBATCH --job-name="fit-jobs"
#SBATCH --time=0-06:00
#SBATCH --mem=20G
#SBATCH --account=def-gflowers
#SBATCH --mail-user=tha111@sfu.ca
#SBATCH --mail-type=FAIL,END


# Don't change anything below this line

autojob.run
