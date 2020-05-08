#!/bin/bash
#SBATCH -n 15
#SBATCH --time=3:00:00

export PATH=$PATH:$HOME/miniconda3/envs/opencv1/bin

python /home/dm1/shikhar/hover_net_modified/src/extract_patches.py