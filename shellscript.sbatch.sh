#!/bin/bash
#SBATCH --job-name=adni_job
#SBATCH --mail-user=torben.sanders@mni.thm.de
#SBATCH --container-image 'nvcr.io/nvidia/tensorflow:23.11-tf2-py3'
#SBATCH --no-container-remap-root
#SBATCH --mem 128G
#SBATCH --cpus-per-task 2
#SBATCH --gpus 1
#SBATCH --output=output_%j.txt

python3 ~/adni/hpc_main.py