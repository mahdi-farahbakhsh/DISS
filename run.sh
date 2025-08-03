#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=get-text-alignment-rewards     #Set the job name
#SBATCH --time=03:00:00            #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                 #Request 1 task
#SBATCH --ntasks-per-node=1        #Request 1 task/core per node
#SBATCH --mem=32G               #Request 32GB per node
#SBATCH --gres=gpu:a100:1     #Request 1 GPU
#SBATCH --output=logs/get-text-alignment-rewards.%j  #Output file name stdout to [JobID]


cd $SCRATCH/semiblind-dps/DISS/
ml Miniconda3
module load WebProxy

source activate /scratch/user/vishnukunde/.conda/envs/DISS

export PYTHONPATH=integrations/dps/diffusion-posterior-sampling:third_party/AdaFace
echo $PYTHONPATH

python3 get_text_alignment_rewards.py