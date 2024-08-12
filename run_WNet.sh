#!/bin/bash
   
#SBATCH --account=ucb520_asc1 # To use additional resources
#SBATCH --time=15:30:00 # 24:00:00
#SBATCH --output=../Jobs/Job-%j.out
#SBATCH --nodes=1           # number of nodes to request  
#SBATCH --mem=80G   #160G          # memory to request
#SBATCH --partition=amilan  # amilan for cpu, aa100 for gpu
##SBATCH --gres=gpu:1       # num GPU to request

gpu=False

module load anaconda/2020.11
conda activate torchenv

cd Solar_Segmentation/

while getopts "f:" flag; do
 case $flag in
   f) expfile=$OPTARG;;
 esac
done

echo "Running experiment with expfile $expfile"
python run_WNet.py -gpu $gpu -f $expfile

#####
# run from ../ with 'sbatch Solar_Segmentation/run_WNet.sh -f exp_file.json'
######