#!/bin/bash
#SBATCH --job-name=DPS         # Job name
#SBATCH --mail-type=BEGIN,END,FAIL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --ntasks=8                      # Run on a 8 cpus (max)
#SBATCH --gres=gpu:tesla:1              # Run on a single GPU (max)
#SBATCH --partition=gpu-research                 # Select GPU Partition
#SBATCH --qos=olympus-research-gpu          # Specify GPU queue
#SBATCH --time=72:00:00                 # Time limit hrs:min:sec current 5 min - 36 hour max

# use the sbatch command to submit your job to the cluster.
# sbatch train.sh

# select your singularity shell (currently cuda10.2-cudnn7-py36)
singularity shell /mnt/lab_files/ECEN403-404/containers/cuda_10.2-cudnn7-py36.sif
# source your virtual environmnet
cd /mnt/shared-scratch/Narayanan_K/mahdi.farahbakhsh/DISS/integrations/daps
source activate DAPS

# Set PYTHONPATH directly
export PYTHONPATH=DAPS:../../../DISS:../../../DISS/third_party/AdaFace

python diss_posterior_sample.py \
+model=ffhq256ddpm \
+sampler=edm_daps \
+data=diss \
+task=diss_inpainting \
+reward=face \
sampler.diffusion_scheduler_config.num_steps=5 \
sampler.annealing_scheduler_config.num_steps=200 \
data.start_id=0 data.end_id=2 name=inpainting_test