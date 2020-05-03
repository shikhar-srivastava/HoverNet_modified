#!/bin/bash
#SBATCH -n 40
#SBATCH --gres=gpu:v100:4
#SBATCH --time=10:00:00

export PATH=$PATH:$HOME/miniconda3/envs/opencv1/bin

python /home/dm1/shikhar/hover_net_modified/src/train.py --gpu='0,1,2,3'