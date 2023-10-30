#!/bin/bash
#SBATCH --job-name=split
#SBATCH --output=job_output_%j.txt
#SBATCH --error=job_error_%j.txt
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --mem=500G
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --partition=cbmm


## train
# python -m torch.distributed.launch --nproc_per_node 2 --use_env --master_port 29301 main.py \
# --dataset cifar10 \
# --epoch 200 \
# --classes 10 \
# --image_size 32 \
# --patch_size 4 \
# --depth 4 \
# --heads 8 \
# --train_batch 512 \
# --lr 0.0005 \
# --hyperbf

#vis.
python -m torch.distributed.launch --nproc_per_node 2 --use_env --master_port 29301 main.py \
--dataset cifar10 \
--epoch 200 \
--classes 10 \
--image_size 32 \
--patch_size 4 \
--depth 4 \
--heads 8 \
--train_batch 512 \
--lr 0.0005 \
--hyperbf \
--vis \
--test \

