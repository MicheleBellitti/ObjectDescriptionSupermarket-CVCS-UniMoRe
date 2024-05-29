#!/bin/bash
#SBATCH --partition=all_usr_prod
#SBATCH --account=cvcs_2023_group23
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4   # Number of tasks (GPUs) per node
#SBATCH --nodes=1             # Number of nodes requested
#SBATCH --output=./logs/frcnn_%j.out
#SBATCH --error=./logs/frcnn_%j.err
#SBATCH --mail-user=319399@studenti.unimore.it
#SBATCH --mail-type=ALL

# Train frcnn
if test $(python3 get_last_epoch.py checkpoints/frcnn/checkpoint.pth) -ge 2
then
    python -u -m torch.distributed.run \
    --nproc_per_node=4 \
    --nnodes=${SLURM_NNODES} \
    --node_rank=${SLURM_NODEID} \
    --master_addr=$(hostname -s) \
    --master_port=12346 \
    train.py --config config.yaml --log_dir logs --model frcnn --verbose --resume_checkpoint checkpoints/frcnn/checkpoint.pth
else
    python -u -m torch.distributed.run \
    --nproc_per_node=4 \
    --nnodes=${SLURM_NNODES} \
    --node_rank=${SLURM_NODEID} \
    --master_addr=$(hostname -s) \
    --master_port=12346 \
    train.py --config config.yaml --log_dir logs --model frcnn --verbose
fi

# Test

python -u -m torch.distributed.run \
    --nproc_per_node=4 \
    --nnodes=${SLURM_NNODES} \
    --node_rank=${SLURM_NODEID} \
    --master_addr=$(hostname -s) \
    --master_port=12347 \
    evaluate.py --config ./config.yaml --model frcnn > logs/frcnn/test.log

#partition can be all_usr_prod or all_serial