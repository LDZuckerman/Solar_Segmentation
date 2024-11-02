#!/bin/bash
   
#SBATCH --account=ucb520_asc1 # To use additional resources
#SBATCH --output=../Jobs/Job-%j.out
#SBATCH --nodes=1           # number of nodes to request  
#SBATCH --gres=gpu:1       # num GPU to request

# Testing mode
#SBATCH --time=1:00:00 # 24:00:00
#SBATCH --ntasks=16           # number of nodes to request  
#SBATCH --partition=atesting_mi100  # amilan for cpu, aa100 for gpu
#SBATCH --qos=testing

## Normal mode
##SBATCH --time=2:00:00 # 24:00:00
##SBATCH --ntasks=20           # number of nodes to request  
##SBATCH --partition=ami100  # amilan for cpu, aa100 for gpu
##SBATCH --qos=normal




gpu=True

module purge
module load rocm/6.1
module load mambaforge/23.1.0-1
mamba activate pytorch241_rocm61
export PYTHONNOUSERSITE=1

cd Solar_Segmentation/

while getopts "f:" flag; do
 case $flag in
   f) expfile=$OPTARG;;
 esac
done

echo "Running experiment with expfile $expfile"
python run_WNet.py -gpu $gpu -f $expfile

#####
# run from ../ with 'sbatch Solar_Segmentation/run_WNet.sh -f wnet_exp_file.json'
######
