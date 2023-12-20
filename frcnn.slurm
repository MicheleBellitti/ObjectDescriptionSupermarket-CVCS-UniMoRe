#!/bin/bash
#SBATCH --partition=students-prod
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/frcnn_%j.log
#SBATCH --error=./logs/frcnn_%j.err


TARGET_EPOCHS=5


# Train frcnn
python3 train_frcnn.py --config ./config.yaml --log_dir ./logs --batch_size 8 --learning_rate 0.002 --num_epochs $TARGET_EPOCHS --verbose


# Test frcnn
python3 test.py --config ./config.yaml --model frcnn
