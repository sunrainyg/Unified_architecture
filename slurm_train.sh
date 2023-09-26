#!/bin/bash
#SBATCH --job-name=split
#SBATCH --output=job_output_%j.txt
#SBATCH --error=job_error_%j.txt
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --mem=150G
#SBATCH --cpus-per-task=16
#SBATCH --time=96:00:00
#SBATCH --partition=cbmm

python -m torch.distributed.launch --nproc_per_node 4 --use_env --master_port 22301 main.py \
--dataset imagenet \
--epoch 100 \
--classes 1000 \
--image_size 224 \
--patch_size 16 \
--train_batch 512 \
--lr 0.0005 \
--hyperbf \

