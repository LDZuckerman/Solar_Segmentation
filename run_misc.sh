#!/bin/bash
   
#SBATCH --account=ucb520_asc1 # To use additional resources
#SBATCH --time=12:00:00
#SBATCH --output=../Jobs/Job-%j.out
#SBATCH --nodes=1           # number of nodes to request  
#SBATCH --mem=160G    # memory to request

module load anaconda/2020.11
conda activate torchenv

cd Solar_Segmentation/

while getopts "t:" flag; do
 case $flag in
   t) task=$OPTARG;;
 esac
done

echo "Running task $task"
python run_misc.py -task $task

#####
# run from ../ with 'sbatch Solar_Segmentation/run_misc.sh -t re-test'
#                   'sbatch Solar_Segmentation/run_misc.sh -t find_image'
######