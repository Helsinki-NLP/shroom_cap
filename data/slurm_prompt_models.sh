#!/bin/bash
#SBATCH --job-name=prompt_models
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1,nvme:10
#SBATCH --mem=56G
#SBATCH --account=project_2001970
#SBATCH --time=04:15:00
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.log

module purge
module load pytorch/2.5
cd /scratch/project_2001194/jrvc/shroom-cap/data

python3 prompt_models.py SPANISH