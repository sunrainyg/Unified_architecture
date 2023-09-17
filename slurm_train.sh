#!/bin/bash
#SBATCH --job-name=split
#SBATCH --output=job_output_%j.txt
#SBATCH --error=job_error_%j.txt
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --partition=cbmm

python main.py \
--dataset cifar10 \
--epoch 100 \
--classes 10 \
--patch_size 4 \
--hyperbf \

