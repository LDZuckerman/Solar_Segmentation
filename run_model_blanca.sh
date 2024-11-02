#!/bin/bash
   
#SBATCH --qos=blanca-nso
#SBATCH --time=15:00:00
#SBATCH --output=../Jobs/Job-%j.out
#SBATCH --nodes=1           # number of nodes to request  
#SBATCH --mem=80G   #160G          # memory to request
##SBATCH --gres=gpu:1       # num GPU to request
gpu=False

module load anaconda #/2020.1s1
conda activate torchenv_new

cd Solar_Segmentation/

while getopts "f:" flag; do
 case $flag in
   f) expfile=$OPTARG;;
 esac
done

echo "Running experiment with expfile $expfile"
python run_model.py -gpu $gpu -f $expfile

#####
# run from ../ with 'sbatch Solar_Segmentation/run_model.sh -f net0_exp_file.json'
######