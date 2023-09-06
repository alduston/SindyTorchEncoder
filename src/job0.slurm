#!/bin/bash

#SBATCH --job-name=job0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ald6fd

#SBATCH --account=dynamicsai
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gpus=1
#SBATCH --time=00-36:00:00

#SBATCH --export=all
#SBATCH --output=./job_outputs/job0.out
#SBATCH --error=./job_outputs/job0.err
# Modules to use (optional).

# Your programs to run.
source activate /mmfs1/gscratch/dynamicsai/ald6fd/SindyTorch
python ensemble_test.py