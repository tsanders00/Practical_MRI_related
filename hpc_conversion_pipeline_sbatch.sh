#!/bin/bash
#SBATCH --job-name=conversion_pipe
#SBATCH --mail-user=torben.sanders@mni.thm.de
#SBATCH --container-image 'nvcr.io/nvidia/tensorflow:23.11-tf2-py3'
#SBATCH --no-container-remap-root
#SBATCH --gpus 1
#SBATCH --cpus-per-task 4
#SBATCH --output=output_%j.txt

python3 ~/adni/hpc_conversion_pipeline.py