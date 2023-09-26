#!/bin/bash
#SBATCH --job-name=split
#SBATCH --output=job_output_%j.txt
#SBATCH --error=job_error_%j.txt
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --partition=cbmm


python cifar_OE.py \
-d cifar100 \
--arch resnet_ensemble \
--epochs 200 \
--train-batch 128 \
--checkpoint /om2/group/cbmm/data/log/num8_paramgroup1/ \
--depth 18 \
--num 4 \
--resume /om2/group/cbmm/data/log/num8_paramgroup1/model_best.pth.tar \
--evaluate 