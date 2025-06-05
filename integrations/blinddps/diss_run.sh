#!/bin/bash
#SBATCH --job-name=Blind-DPS         # Job name
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
cd /mnt/shared-scratch/Narayanan_K/mahdi.farahbakhsh/DISS/integrations/blinddps
source activate DPS_paper

# Set PYTHONPATH directly
export PYTHONPATH=blind-dps:../../../DISS:../../../DISS/third_party/AdaFace

python3 diss_deblur.py \
    --img_model_config=blind-dps/configs/model_config.yaml \
    --kernel_model_config=blind-dps/configs/kernel_model_config.yaml \
    --diffusion_config=diss_configs/diss_diffusion_config.yaml \
    --task_config=diss_configs/diss_motion_deblur.yaml \
    --reg_ord=1 \
    --reg_scale=1.0 \
    --path=test
