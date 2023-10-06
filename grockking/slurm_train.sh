#!/bin/bash
#SBATCH --job-name=split
#SBATCH --output=logs/job_output_%j.txt
#SBATCH --error=logs/job_error_%j.txt
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=cbmm

python grokking_main.py