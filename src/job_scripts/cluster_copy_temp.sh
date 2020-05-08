#!/bin/bash
#SBATCH -n 10
#SBATCH --gres=gpu:v100:1
#SBATCH --time=20:00:00
echo 'Trying'
cp -r /tmp//v1.0 /home/dm1/shikhar/hover_net_modified
echo 'Done'