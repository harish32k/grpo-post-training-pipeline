#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=08:00:00
#SBATCH --job-name=g1_v100
#SBATCH --output=batch_scripts/run_logs/m1/model_1.%j.out
#SBATCH --error=batch_scripts/run_logs/m1/model_1.%j.out

lscpu
nvidia-smi
module load anaconda3
source activate llmrl

python run_grpo.py
