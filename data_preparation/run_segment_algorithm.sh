#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --partition=amilan
#SBATCH --output=Job-%j.out

module purge

module load anaconda
module load python

conda activate myenv

cd /projects/lezu7058/Solar_Segmentation/
python run_segment_algorithm.py
