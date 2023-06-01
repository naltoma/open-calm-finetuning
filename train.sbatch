#!/bin/bash
#SBATCH --job-name open-calm3b
#SBATCH --output logs/%x-%j.log
#SBATCH --error logs/%x-%j.err
#SBATCH --nodes 1
#SBATCH --gpus tesla:1

date
singularity exec --nv huggingface.sif python3 opencalm_7b_lora_train.py
date
