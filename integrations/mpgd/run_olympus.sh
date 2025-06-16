#!/bin/bash
#SBATCH --job-name=dm-lin-inv      # Job name
#SBATCH --mail-type=BEGIN,END,FAIL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=vishnukunde@tamu.edu  #Where to send mail    
#SBATCH --ntasks=1                      # Run on a 8 cpus (max)
#SBATCH --gres=gpu:a100:1              # Run on a single GPU (max)
#SBATCH --partition=gpu-research                 # Select GPU Partition
#SBATCH --qos=olympus-research-gpu          # Specify GPU queue
#SBATCH --time=00:30:00                 # Time limit hrs:min:sec current 5 min - 36 hour max
#SBATCH --output=logs/%x_%j.out        # Standard output and error log

# select your singularity shell (currently cuda10.2-cudnn7-py36)
singularity shell /mnt/lab_files/ECEN403-404/containers/cuda_10.2-cudnn7-py36.sif

python run_inverse.py \
                --model_config=configs/model_config.yaml \
                --diffusion_config=configs/mpgd_diffusion_search_config.yaml \
                --task_config=configs/super_resolution_4x_config_full_images.yaml \
                --reward_eval_config=configs/reward_adaface.yaml \
                --search_algo_config=configs/search_group.yaml \
                --timestep=100 \
                --scale=4 \
                --method="mpgd_wo_proj" \
                --save_dir='./outputs_test_mpgd/' \
                --n_images=70 \
                --temp=0.05 \
                --num_particles=8 \
                --batch_size=8 \
                --resample_rate=8 \
                --ref_faces_path='./data/additional_images/' \