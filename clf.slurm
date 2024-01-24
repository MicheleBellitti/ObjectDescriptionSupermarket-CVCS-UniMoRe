#!/bin/bash
#SBATCH --partition=all_serial
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1

#SBATCH --nodes=1             # Number of nodes requested
#SBATCH --output=./logs/clf_%j.out
#SBATCH --error=./logs/clf_%j.err

#SBATCH --mail-user=274236@studenti.unimore.it
#SBATCH --mail-type=ALL


# train
python3 img_classification.py