#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=dm-mpgd     #Set the job name to "JobExample1"
#SBATCH --time=03:00:00            #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                 #Request 1 task
#SBATCH --ntasks-per-node=1        #Request 1 task/core per node
#SBATCH --mem=32G               #Request 64GB per node
##SBATCH --gres=gpu:a100:1     #Request 1 GPU
#SBATCH --output=logs/search-for-inv.%j  #Output file name stdout to [JobID]


cd $SCRATCH/semiblind-dps/DISS/integrations
ml Miniconda3
module load WebProxy

source activate /scratch/user/vishnukunde/.conda/envs/DISS

python3 add_inits.py