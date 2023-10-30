#!/bin/bash
#SBATCH --job-name=split
#SBATCH --output=job_output_%j.txt
#SBATCH --error=job_error_%j.txt
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=16
#SBATCH --time=96:00:00
#SBATCH --partition=cbmm

export CUDA_HOME=/cm/shared/modulefiles/cuda70/blas/7.0.28

python dissect_classifier.py