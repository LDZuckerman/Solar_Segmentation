#!/bin/bash
   
#SBATCH --account=ucb520_asc1 # To use additional resources
#SBATCH --time=12:00:00
#SBATCH --output=../Jobs/Job-%j.out
#SBATCH --nodes=1           # number of nodes to request  
#SBATCH --mem=160G    # memory to request

module load anaconda/2020.11
conda activate torchenv_new

cd Solar_Segmentation/

task_flag='None'
while getopts "t:f:" flag; do
 case $flag in
   t) task=$OPTARG;;
   f) task_flag=$OPTARG;;
 esac
done

echo "Running task $task with flag $task_flag"
python run_misc.py -task $task -flag $task_flag

#####
# run from ../ with 'sbatch Solar_Segmentation/run_misc.sh -t re-test'
#                   'sbatch Solar_Segmentation/run_misc.sh -t find_image'
######