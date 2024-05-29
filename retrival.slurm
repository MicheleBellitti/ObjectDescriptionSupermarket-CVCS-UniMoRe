#!/bin/bash
#SBATCH --partition=all_serial
#SBATCH --account=cvcs_2023_group23
#SBATCH --time=00:02:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1   # Number of tasks (GPUs) per node
#SBATCH --nodes=1             # Number of nodes requested
#SBATCH --output=./logs/retrival_%j_output.out
#SBATCH --error=./logs/retrival_%j_error.err
#SBATCH --mail-user=319399@studenti.unimore.it
#SBATCH --mail-type=ALL

# Retrival
python -u retrival.py