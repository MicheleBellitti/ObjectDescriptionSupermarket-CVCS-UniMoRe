import pytest
from unittest.mock import MagicMock
from train import *
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
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.detection import _utils as det_utils
import torchvision.ops as ops
from torch.nn import SmoothL1Loss
from torch.utils.tensorboard import SummaryWriter
from datasets import SKUDataset, TEST_TRANSFORM, TRAIN_TRANSFORM, VAL_TRANSFORM, custom_collate_fn
import yaml
from functools import partial
from model import CustomRetinaNet
from rich.progress import Progress, BarColumn, TimeRemainingColumn

# Constants for tests
MODEL_NAMES = ['ssd', 'retinanet', 'frcnn']
NUM_CLASSES = [2, 10, 100]
LOG_DIRS = ['/tmp/logs', '/var/log/train']
VERBOSE_FLAGS = [True, False]
DEVICES = ['cpu', 'cuda']
EPOCHS = [0, 1, 5]
PRINT_FREQS = [10, 100, 1000]
BATCH_SIZES = [1, 4, 16]
LEARNING_RATES = [0.001, 0.01, 0.1]
STEP_SIZES = [3, 5, 10]
GAMMAS = [0.1, 0.5, 0.9]

# Fixtures for common setup
@pytest.fixture
def mock_device(monkeypatch):
    monkeypatch.setattr(torch, "cuda", MagicMock(is_available=lambda: True))
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def mock_dataloader():
    dataset = SKUDataset(split="train", transform=TRAIN_TRANSFORM)
    return DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)

@pytest.fixture
def mock_model():
    return CustomRetinaNet(num_classes=2)

@pytest.fixture
def mock_optimizer(mock_model):
    return torch.optim.AdamW(mock_model.parameters(), lr=0.001)

@pytest.fixture
def mock_writer(tmp_path):
    return SummaryWriter(log_dir=str(tmp_path))

# Parametrized tests
@pytest.mark.parametrize("model_name,num_classes", [(m, n) for m in MODEL_NAMES for n in NUM_CLASSES])
def test_get_model(model_name, num_classes):
    # Act
    model = get_model(model_name, num_classes)

    # Assert
    assert isinstance(model, torch.nn.Module), f"Failed for model: {model_name} with num_classes: {num_classes}"

@pytest.mark.parametrize("log_dir,verbose", [(l, v) for l in LOG_DIRS for v in VERBOSE_FLAGS])
def test_setup_logging(log_dir, verbose):
    # Arrange
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Act
    setup_logging(log_dir, verbose)

    # Assert
    assert os.path.isfile(os.path.join(log_dir, 'train.log')), f"Log file not created in {log_dir} with verbose={verbose}"

@pytest.mark.parametrize("device,epoch,print_freq", [(d, e, p) for d in DEVICES for e in EPOCHS for p in PRINT_FREQS])
def test_train_one_epoch(
    model, mock_optimizer, mock_dataloader, mock_writer, device, epoch, print_freq
):
    # Arrange
    device = torch.device(device)
    model.to(device)

    # Act
    avg_loss = train_one_epoch(
        model, mock_optimizer, mock_dataloader, device, epoch, print_freq, mock_writer
    )

    # Assert
    assert isinstance(
        avg_loss, float
    ), f"Average loss is not a float for device: {device}, epoch: {epoch}, print_freq: {print_freq}"


@pytest.mark.parametrize("device,epoch", [(d, e) for d in DEVICES for e in EPOCHS])
def test_validate(mock_model, mock_dataloader, mock_writer, device, epoch):
    # Arrange
    device = torch.device(device)
    mock_model.to(device)

    # Act
    avg_val_loss = validate(mock_model, mock_dataloader, device, epoch, mock_writer)

    # Assert
    assert isinstance(avg_val_loss, float), f"Average validation loss is not a float for device: {device}, epoch: {epoch}"

@pytest.mark.parametrize("batch_size,lr,step_size,gamma", [(b, lr, s, g) for b in BATCH_SIZES for lr in LEARNING_RATES for s in STEP_SIZES for g in GAMMAS])
def test_main_integration(batch_size, lr, step_size, gamma, tmp_path, monkeypatch):
    # Arrange
    config = {
        "batch_size": batch_size,
        "lr": lr,
        "epochs": 1,
        "print_freq": 100,
        "step_size": step_size,
        "gamma": gamma,
    }
    args = argparse.Namespace(
        config=str(tmp_path / "config.yaml"),
        model="retinanet",
        log_dir=str(tmp_path / "logs"),
        resume_checkpoint=None,
        verbose=False,
    )
    with open(args.config, "w") as f:
        yaml.dump(config, f)



    monkeypatch.setattr("train_one_epoch", MagicMock(return_value=0.0))
    monkeypatch.setattr("validate", MagicMock(return_value=0.0))
    monkeypatch.setattr("torch.save", MagicMock())

    # Act
    main()

    # Assert
    # Check if the mocked functions were called
    train_one_epoch.assert_called()
    validate.assert_called()
    torch.save.assert_called()
