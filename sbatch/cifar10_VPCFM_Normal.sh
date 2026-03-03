#!/bin/bash

#SBATCH --account=nlp
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --job-name=houjun-cifar10_VPCFM_Normal
#SBATCH --mem=32G
#SBATCH --open-mode=append
#SBATCH --output=./logs/cifar10_VPCFM_Normal.log
#SBATCH --partition=jag-standard
#SBATCH --time=14-0

cd .

uv python install 3.11 --force
/sailhome/houjun/bin/develop "source .venv/bin/activate && python train_slurm.py --base Normal --flow VPCFM --dataset cifar10 --epochs 100"
