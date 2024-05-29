#!/bin/bash
#SBATCH --partition=all_serial
#SBATCH --account=cvcs_2023_group23
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1          # (Can be 1,3 -> change nproc_per_node accordingly)
#SBATCH --ntasks-per-node=1   # Number of tasks (GPUs) per node (Can be 1,3)
#SBATCH --nodes=1             # Number of nodes requested
#SBATCH --output=./logs/output_%j.out
#SBATCH --error=./logs/output_%j.err

python -u -m torch.distributed.run \
--nproc_per_node=1 \
--nnodes=${SLURM_NNODES} \
--node_rank=${SLURM_NODEID} \
--master_addr=$(hostname -s) \
--master_port=12346 \
inference_frcnn.py

#all_usr_prod or all_serial