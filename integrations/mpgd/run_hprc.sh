#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=dm-mpgd     #Set the job name to "JobExample1"
#SBATCH --time=03:00:00            #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                 #Request 1 task
#SBATCH --ntasks-per-node=1        #Request 1 task/core per node
#SBATCH --mem=32G               #Request 64GB per node
#SBATCH --gres=gpu:a100:1     #Request 1 GPU
#SBATCH --output=logs/search-for-inv.%j  #Output file name stdout to [JobID]


cd $SCRATCH/semiblind-dps/DISS/integrations/mpgd
ml Miniconda3
module load WebProxy

source activate mpgd


python run_inverse_diss.py \
        --model_config=diss_configs/model_config.yaml \
        --diffusion_config=diss_configs/mpgd_diffusion_search_config.yaml \
        --task_config=diss_configs/diss_super_resolution_config.yaml \
        --timestep=100 \
        --scale=4 \
        --method="mpgd_wo_proj" \
        --save_dir='./outputs_test_mpgd_corrected/' \
        --n_images=70 \
        --temp=0.05  \
        --resample_rate=8 \
        --num_particles=8 \
        --batch_size=8 \
        --ref_faces_path='../../data/additional_images/' 