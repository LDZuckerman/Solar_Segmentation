#!/bin/bash
   
#SBATCH --account=ucb520_asc1 # To use additional resources
#SBATCH --time=5:00:00
#SBATCH --output=../../../Jobs/Job-%j.out
#SBATCH --nodes=1           # number of nodes to request  
#SBATCH --mem=50G           # memory to request
#SBATCH --partition=amilan  # amilan for cpu, aa100 for gpu
##SBATCH --gres=gpu:1       # num GPU to request
gpu=False

module load anaconda/2020.11
conda activate torchenv_new

echo "Running norm.py"
python norm.py

