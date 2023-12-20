#!/bin/bash
#SBATCH --partition=all_usr_prod
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:3
#SBATCH --ntasks-per-node=3   # Number of tasks (GPUs) per node
#SBATCH --nodes=1             # Number of nodes requested
#SBATCH --output=./logs/ssd_%j.out
#SBATCH --error=./logs/ssd_%j.err


# Train ssd
if test $(python3 get_last_epoch.py checkpoints/ssd/checkpoint.pth) -ge 2
then
    python -m torch.distributed.run \
    --nproc_per_node=3 \
    --nnodes=${SLURM_NNODES} \
    --node_rank=${SLURM_NODEID} \
    --master_addr=$(hostname -s) \
    --master_port=12346 \
    train.py --config config.yaml --log_dir logs --model ssd --verbose --resume_checkpoint checkpoints/ssd/checkpoint.pth
else
    python -m torch.distributed.run \
    --nproc_per_node=3 \
    --nnodes=${SLURM_NNODES} \
    --node_rank=${SLURM_NODEID} \
    --master_addr=$(hostname -s) \
    --master_port=12346 \
    train.py --config config.yaml --log_dir logs --model ssd --verbose
fi
# Test

python -m torch.distributed.run \
    --nproc_per_node=3 \
    --nnodes=${SLURM_NNODES} \
    --node_rank=${SLURM_NODEID} \
    --master_addr=$(hostname -s) \
    --master_port=12347 \
    evaluate.py --config ./config.yaml --model ssd
