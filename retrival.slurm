#!/bin/bash
#SBATCH --partition=all_serial
#SBATCH --time=00:10:00
#SBATCH --output=./logs/retrival_%j_output.out
#SBATCH --error=./logs/retrival_%j_error.err
#SBATCH --mail-user=319399@studenti.unimore.it
#SBATCH --mail-type=ALL

# Retrival
python -u retrival.py

#partition can be all_usr_prod or all_serial