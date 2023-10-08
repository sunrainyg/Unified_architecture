#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/job_output_%j.txt
#SBATCH --error=logs/job_error_%j.txt
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --partition=cbmm

python grokking_main.py