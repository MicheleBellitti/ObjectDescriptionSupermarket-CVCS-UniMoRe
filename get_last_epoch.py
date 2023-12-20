# get_last_epoch.py
import torch
import sys
import os

def get_last_epoch(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    return checkpoint['epoch'] + 1

if __name__ == '__main__':
    epoch = get_last_epoch(sys.argv[1]) if os.path.exists(sys.argv[1]) else 1
    print(epoch)
