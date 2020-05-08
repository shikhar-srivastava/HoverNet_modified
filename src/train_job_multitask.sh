#!/bin/bash
#SBATCH -n 80
#SBATCH --gres=gpu:v100:8
#SBATCH --time=20:00:00

echo 'd' | /home/dm1/miniconda3/envs/opencv1/bin/python -u /home/dm1/shikhar/hover_net_modified/src/train_multitask.py --gpu='0,1,2,3,4,5,6,7'