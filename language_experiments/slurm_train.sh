#!/bin/bash
#SBATCH --job-name=split
#SBATCH --output=job_output_%j.txt
#SBATCH --error=job_error_%j.txt
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --mem=500G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --partition=cbmm

python 1022.py