import torch
import os
import argparse
import logging
import time
import math
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import _utils as det_utils
import torchvision.ops as ops
from torch.nn import SmoothL1Loss
from torch.utils.tensorboard import SummaryWriter
from datasets import SKUDataset, TEST_TRANSFORM, TRAIN_TRANSFORM, VAL_TRANSFORM, custom_collate_fn
import yaml
from functools import partial
import random

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True, trust_repo=True)

# Images
train_dataset = SKUDataset(split='train', transform=TRAIN_TRANSFORM)
val_dataset = SKUDataset(split='val', transform=VAL_TRANSFORM)
# imgs = val_dataset.image_names[:10] # batch of images
imgs = [
        f'/work/cvcs_2023_group23/SKU110K_fixed/images/test_{int(random.uniform(0, 2940))}.jpg',
        f'/work/cvcs_2023_group23/SKU110K_fixed/images/test_{int(random.uniform(0, 2940))}.jpg',
        f'/work/cvcs_2023_group23/SKU110K_fixed/images/test_{int(random.uniform(0, 2940))}.jpg'
        ]  # batch of images


# Inference
results = model(imgs)

# Results
results.print()

results.save('train_results/yolo')