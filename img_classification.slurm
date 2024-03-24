#!/bin/bash
#SBATCH --partition=all_serial
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1

#SBATCH --nodes=1             # Number of nodes requested
#SBATCH --output=./logs/clf_%j.out
#SBATCH --error=./logs/clf_%j.err

#SBATCH --mail-user=319399@studenti.unimore.it
#SBATCH --mail-type=ALL


# train
python3 img_classification.py
# python3 imagerec_bagging.py

# test
python3 img_classification_test.py