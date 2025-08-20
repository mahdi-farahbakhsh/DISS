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
cd /mnt/shared-scratch/Narayanan_K/mahdi.farahbakhsh/DISS/integrations/dps
source activate DPS_paper

# Set PYTHONPATH directly
export PYTHONPATH=diffusion-posterior-sampling:../../../DISS:../../../DISS/third_party/AdaFace

python3 diss_sample_conditions.py \
    --model_config=diffusion-posterior-sampling/configs/model_config.yaml \
    --diffusion_config=diffusion-posterior-sampling/configs/diffusion_config.yaml \
    --task_config=diss_configs/faceID/diss_inpainting_config.yaml \
    --path=face_box \
    --metrics=psnr,lpips,ssim,adaface

python3 diss_sample_conditions.py \
    --model_config=diffusion-posterior-sampling/configs/model_config.yaml \
    --diffusion_config=diffusion-posterior-sampling/configs/diffusion_config.yaml \
    --task_config=diss_configs/faceID/diss_nonlinear_deblur_config.yaml \
    --path=face_nonlinear \
    --metrics=psnr,lpips,ssim,adaface

python3 diss_sample_conditions.py \
    --model_config=diffusion-posterior-sampling/configs/model_config.yaml \
    --diffusion_config=diffusion-posterior-sampling/configs/diffusion_config.yaml \
    --task_config=diss_configs/faceID/diss_super_resolution_config.yaml \
    --path=face_super \
    --metrics=psnr,lpips,ssim,adaface

python3 diss_sample_conditions.py \
    --model_config=diffusion-posterior-sampling/configs/model_config.yaml \
    --diffusion_config=diffusion-posterior-sampling/configs/diffusion_config.yaml \
    --task_config=diss_configs/faceID/diss_motion_deblur_config.yaml \
    --path=face_motion \
    --metrics=psnr,lpips,ssim,adaface

python3 diss_sample_conditions.py \
    --model_config=diffusion-posterior-sampling/configs/model_config.yaml \
    --diffusion_config=diffusion-posterior-sampling/configs/diffusion_config.yaml \
    --task_config=diss_configs/faceID/diss_gaussian_deblur_config.yaml \
    --path=face_gaussian \
    --metrics=psnr,lpips,ssim,adaface