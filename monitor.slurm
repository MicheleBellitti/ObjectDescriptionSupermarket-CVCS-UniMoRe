#!/bin/bash
#SBATCH --job-name='monitor_retina'
#SBATCH --partition=all_usr_prod
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00  # Adjust as needed

python bot.py
