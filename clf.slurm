#!/bin/bash
#SBATCH --partition=all_usr_prod
#SBATCH --account=ai4bio2023
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1   # Number of tasks (GPUs) per node
#SBATCH --nodes=1             # Number of nodes requested
#SBATCH --output=./logs/clf_densenet121/clf_%j.out
#SBATCH --error=./logs/clf_densenet121/clf_%j.err
#SBATCH --mail-user=319399@studenti.unimore.it
#SBATCH --mail-type=ALL

# Train
python -u clf_train.py --checkpoint_path ./checkpoints/clf_densetnet121/240324/40Epochs/last.ckpt

# Test
python -u clf_test.py --model_checkpoint ./checkpoints/clf_densetnet121/240324/40Epochs/last.ckpt --output_dir ./train_results/clf_densenet121/

# Eval
python -u clf_evaluate.py --predictions_path ./train_results/clf_densenet121/inference_results.json