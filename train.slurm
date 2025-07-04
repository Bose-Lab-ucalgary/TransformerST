#!/bin/bash

####### Reserve computing resources #############
#SBATCH --job-name=transformerST_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100

####### Run your script #########################
source ~/software/init-conda
conda activate transformerst-env


python HER2_histology.py