#!/bin/bash
#SBATCH --job-name='bot'
#SBATCH --partition=all_serial
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00  # Adjust as needed

python bot.py
