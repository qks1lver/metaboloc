#!/usr/bin/env bash

#SBATCH -J metaboloc
#SBATCH -p SHARED
#SBATCH -N 1
#SBATCH -c 24
#SBATCH --mem=10000

module purge; module load Python/3.6.0
srun python3 metaboloc.py "$@" 2>&1
