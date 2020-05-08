#!/bin/bash
#SBATCH -n 10
#SBATCH --gres=gpu:v100:1
#SBATCH --time=20:00:00

/home/dm1/miniconda3/envs/opencv1/bin/python -u /home/dm1/shikhar/hover_net_modified/src/infer.py --gpu='0'


# PROCESS

# /home/dm1/miniconda3/envs/opencv1/bin/python -u /home/dm1/shikhar/hover_net_modified/src/process.py

# Compute Statistics (Use Processed Predictions for computing statistics)

# /home/dm1/miniconda3/envs/opencv1/bin/python -u /home/dm1/shikhar/hover_net_modified/src/compute_stats.py --pred_dir='/home/dm1/shikhar/hover_net_modified/MoNuSAC_processed/Overlay/v1.0/np_hv_proc/' --true_dir='/home/dm1/shikhar/hover_net_modified/MoNuSAC_processed/Valid_Labels/'