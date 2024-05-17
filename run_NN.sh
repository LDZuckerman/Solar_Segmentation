#!/bin/bash

#SBATCH --nodes=1           # would I need to use multithreading to change this?
#SBATCH --ntasks=3          # would I need to use multithreading to change this?
#SBATCH --time=08:00:00
#SBATCH --partition=aa100   # default amilan doesn't have GPU? Using high-mem 'amem' gives sbatch: error: Batch job submission failed: Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)
#SBATCH --gres=gpu:2        # add this when using parition with GPU
#SBATCH --output=../Jobs/Job-%j.out

module purge
module load anaconda
conda activate myenv

cd /projects/lezu7058/solar_ML/Solar_Segmentation/
python run_WNET.py