#!/bin/bash
#SBATCH --partition=all_usr_prod
#SBATCH --constraint="gpu_RTX6000_24G"
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
#SBATCH --ntasks-per-node=4  # Number of tasks (GPUs) per node
#SBATCH --nodes=1             # Number of nodes requested
#SBATCH --output=./logs/retina_%j.out
#SBATCH --error=./logs/retina_%j.err
#SBATCH --job-name=retinaNet

RESOURCES=4
CHECKPOINT_PATH=$(pwd)"checkpoints/retinanet/checkpoint.pth"
# Train
if test $(python3 get_last_epoch.py $CHECKPOINT_PATH) -ge 2
then
    python -m torch.distributed.run \
    --nproc_per_node=$RESOURCES \
    --nnodes=${SLURM_NNODES} \
    --node_rank=${SLURM_NODEID} \
    --master_addr=$(hostname -s) \
    --master_port=12346 \
    train.py --config config.yaml --log_dir logs --model retinanet --verbose --resume_checkpoint $CHECKPOINT_PATH
else
    python -m torch.distributed.run \
    --nproc_per_node=$RESOURCES \
    --nnodes=${SLURM_NNODES} \
    --node_rank=${SLURM_NODEID} \
    --master_addr=$(hostname -s) \
    --master_port=12346 \
    train.py --config config.yaml --log_dir logs --model retinanet --verbose
fi

# Test

python -m torch.distributed.run \
    --nproc_per_node=$RESOURCES \
    --nnodes=${SLURM_NNODES} \
    --node_rank=${SLURM_NODEID} \
    --master_addr=$(hostname -s) \
    --master_port=12347 \
    evaluate.py --config ./config.yaml --model retinanet
