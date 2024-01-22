import os
import argparse
import logging
import time
import math
import torch
from utils import *
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.ops as ops
from torch.nn import SmoothL1Loss
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from datasets import SKUDataset, TEST_TRANSFORM, TRAIN_TRANSFORM, VAL_TRANSFORM, custom_collate_fn
import yaml
from functools import partial
from model import CustomRetinaNet
from rich.progress import Progress, BarColumn, TimeRemainingColumn

from adabelief_pytorch import AdaBelief

# Enable benchmark mode in cudnn
cudnn.benchmark = True

def setup_logging(log_dir, verbose):
    """
    Set up logging for the training process.i

    Args:
        log_dir (str): Directory to save log files.
        verbose (bool): Flag to set verbose logging.
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename=os.path.join(log_dir, 'train.log'), level=logging.INFO if verbose else logging.WARNING, format=log_format)
    '''
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(console_handler)
    '''

def get_model(model_name, num_classes):
    """
    Retrieve the model based on the model name.

    Args:
        model_name (str): Name of the model to use ('ssd', 'retinanet', 'frcnn').
        num_classes (int): Number of classes for the model.

    Returns:
        torch.nn.Module: The requested object detection model.
    """
    if model_name == 'ssd':
        model = ssdlite320_mobilenet_v3_large(weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
        in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
        num_anchors = model.anchor_generator.num_anchors_per_location()
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

        dropout = nn.Dropout(p=0.3)
        model.head.classification_head = nn.Sequential(
        SSDLiteClassificationHead(in_channels, num_anchors, 2, norm_layer),
        dropout
            )
    elif model_name == 'retinanet':
        model = CustomRetinaNet(num_classes=num_classes)
    elif model_name == 'frcnn':
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, writer):
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        data_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        device (torch.device): Device to train the model on.
        epoch (int): Current epoch number.
        print_freq (int): Frequency of printing training status.
    """
    
    total_loss = 0.0
    model.train()
    with Progress("[progress.description]{task.description}", BarColumn(), "[progress.percentage]{task.percentage:>3.0f}%", TimeRemainingColumn(), expand=True) as progress:
        train_task = progress.add_task("[green]Training", total=len(data_loader))

        for batch_idx, (images, x1, y1, x2, y2, class_id, image_width, image_height) in enumerate(data_loader):
            images = images.to(device)


            x1 = x1.to(device)
            y1 = y1.to(device)
            x2 = x2.to(device)
            y2 = y2.to(device)
            class_id = (class_id - class_id.min()).to(device)
            image_width = image_width.to(device)
            image_height = image_height.to(device)

            # print(images.shape)
            # Create a list of target dictionaries
            targets = []
            for i in range(len(images)):
                target = {}
                target['boxes'] = torch.ones((4, x1[i].shape[0]))
                target["boxes"][0] = x1[i]
                target["boxes"][1] = y1[i]
                target["boxes"][2] = x2[i]
                target["boxes"][3] = y2[i]
                target["boxes"] = target["boxes"].T
                target["boxes"] = target["boxes"].to(device)
                target['labels'] = class_id[i].to(device)
                target['image_width'] = image_width[i].to(device)
                target['image_height'] = image_height[i].to(device)
                targets.append(target)


            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            clip_gradient(optimizer, 1.0)
            optimizer.step()

            total_loss += losses.item()
            progress.update(train_task, advance=1, description=f"[green]Training Epoch: {epoch} [{batch_idx}/{len(data_loader)}] Loss: {losses.item():.4f}")

            if batch_idx % print_freq == 0:
                logging.info(f"Epoch: [{epoch}] [{batch_idx}/{len(data_loader)}] Loss: {losses.item()}")

            # Log additional statistics to TensorBoard
            writer.add_scalar('Loss/Train/Batch', losses.item(), epoch * len(data_loader) + batch_idx)
            for key, value in loss_dict.items():
                writer.add_scalar(f'Loss/Train/{key}', value.item(), epoch * len(data_loader) + batch_idx)

    avg_loss = total_loss / len(data_loader)
    writer.add_scalar('Loss/Train/Epoch', avg_loss, epoch)
    return avg_loss

def validate(model, data_loader, device, epoch, writer):
    """
    Validate the model.

    Args:
        model (torch.nn.Module): The model to validate.
        data_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
        device (torch.device): Device to validate the model on.

    Returns:
        float: The average loss of the validation.
    """
    model.eval()
    val_loss = 0.0
    labels_criterion = ops.sigmoid_focal_loss
    boxes_criterion = ops.generalized_box_iou_loss
    with Progress("[progress.description]{task.description}", BarColumn(), "[progress.percentage]{task.percentage:>3.0f}%", TimeRemainingColumn(), expand=True) as progress:
        val_task = progress.add_task("[cyan]Validating", total=len(data_loader))
    
        for batch_idx, (images, x1, y1, x2, y2, class_id, image_width, image_height) in enumerate(data_loader):
            images = images.to(device)
            x1 = x1.to(device)
            y1 = y1.to(device)
            x2 = x2.to(device)
            y2 = y2.to(device)
            class_id = class_id.to(device)
            image_width = image_width.to(device)
            image_height = image_height.to(device)
            

            # Create a list of target dictionaries
            targets = []
            for i in range(len(images)):
                target = {}
                target['boxes'] = torch.ones((4, x1[i].shape[0]))
                target["boxes"][0] = x1[i]
                target["boxes"][1] = y1[i]
                target["boxes"][2] = x2[i]
                target["boxes"][3] = y2[i]
                target["boxes"] = target["boxes"].T
                target["boxes"] = target["boxes"].to(device)
                target['labels'] = class_id[i].to(device)
                target['image_width'] = image_width[i].to(device)
                target['image_height'] = image_height[i].to(device)
                targets.append(target)

            # Forward pass
            outputs = model(images)
            # print(outputs)

            # Check if 'boxes' and 'labels' keys exist in outputs dictionary
            if outputs and 'boxes' in outputs[0] and 'labels' in outputs[0]:
                output_boxes = torch.cat([output['boxes'] for output in outputs])
                output_labels = torch.cat([output['labels'] for output in outputs])
            else:
                print(outputs)
                logging.info("Error: Keys 'boxes' and 'labels' not found in outputs dictionary.")
                raise KeyError("Keys 'boxes' and 'labels' not found in outputs dictionary.")

            # Extract tensors from targets list
            target_boxes = torch.cat([target['boxes'] for target in targets])
            target_labels = torch.cat([target['labels'] for target in targets])
            # print(f"Validation shapes: {output_boxes.shape} {target_boxes.shape}")
            
            # Adjust tensors shapes if needed
            if target_boxes.shape[0] < output_boxes.shape[0]:
                # Extract the first target_boxes.shape[0] elements from output_boxes and output_labels
                output_boxes = output_boxes[:target_boxes.shape[0], :]
                output_labels = output_labels[:target_labels.shape[0]]
                
            elif target_boxes.shape[0] > output_boxes.shape[0]:
                # Extract the first output_boxes.shape[0] elements from target_boxes and target_labels
                target_boxes = target_boxes[:output_boxes.shape[0], :]
                target_labels = target_labels[:output_labels.shape[0]]

            
            print(f"Validation shapes matched: {output_boxes.shape} {target_boxes.shape}")
            output_labels = output_labels.float()  # Convert to float tensor
            target_labels = target_labels.float()
            loss = boxes_criterion(output_boxes, target_boxes, reduction='mean') + labels_criterion(output_labels, target_labels, reduction='mean')

            # Compute validation losss
            
            val_loss += loss.item()
            progress.update(val_task, advance=1, description=f"[cyan]Validating Epoch: {epoch} [{batch_idx}/{len(data_loader)}] Loss: {loss.item():.4f}")

            # Log additional statistics to TensorBoard
            writer.add_scalar('Loss/Val/Batch', loss.item(), epoch * len(data_loader) + batch_idx)

    avg_val_loss = val_loss / len(data_loader)
    # writer.add_scalar('Loss/Val/Epoch', avg_val_loss, epoch)
    return avg_val_loss

def main():
    parser = argparse.ArgumentParser(description="Object Detection Training Script")
    parser.add_argument("--config", required=True, help="Path to the YAML config file")
    parser.add_argument(
        "--model",
        choices=["ssd", "retinanet", "frcnn"],
        required=True,
        help="Model to use for training",
    )
    
    parser.add_argument("--log_dir", required=True, help="Path to the log directory")
    parser.add_argument(
        "--resume_checkpoint", help="Path to checkpoint file to resume training"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    # Load config file
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    # Setup logging
    setup_logging(f'{args.log_dir}/{args.model}', args.verbose)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Load model
    model = get_model(args.model, 2)
    model = model.to(local_rank)
    torch.cuda.empty_cache()
    model = DDP(model, device_ids=[local_rank])

    # Data loading
    train_dataset = SKUDataset(split="train", transform=TRAIN_TRANSFORM)
    val_dataset = SKUDataset(split="val", transform=VAL_TRANSFORM)
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        sampler=train_sampler,
        num_workers=2,
        collate_fn=custom_collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        sampler = val_sampler,
        num_workers=2,
        collate_fn=custom_collate_fn,
    )

    # Optimizer and scheduler
    optimizer = SGD(model.parameters(), lr=config['lr'], weight_decay=config["weight_decay"], momentum=config['momentum'])
    #optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config['weight_decay'])
    #optimizer = AdaBelief(model.parameters(), lr=config['lr'], eps=1e-16, betas=(0.9, 0.999), weight_decay=config['weight_decay'])
    
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1) 
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True, eps=1e-5, cooldown=0, min_lr=0)
    
    # Tensorboard writer
    writer = SummaryWriter(log_dir=args.log_dir,flush_secs=90)

    # Resume from checkpoint
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        logging.info("Resuming from checkpoint at epoch {}".format(start_epoch))
    else:
        start_epoch = 0

    # Training loop
    for epoch in range(start_epoch, config["epochs"]):
        train_sampler.set_epoch(epoch)
        
        train_loss = train_one_epoch(
            model,
            optimizer,
            train_dataloader,
            device,
            epoch,
            config["print_freq"],
            writer,
        )
        writer.add_scalar("Loss/Training", train_loss, epoch)
        val_sampler.set_epoch(epoch)
        val_loss = validate(model, val_dataloader, device, epoch, writer)
        writer.add_scalar("Loss/Validation", val_loss, epoch)

        lr_scheduler.step(train_loss)
        if dist.get_rank() == 0:
        # Save checkpoint
            save_checkpoint(epoch + 1, model, optimizer, lr_scheduler, args.model)

    writer.close()

if __name__ == "__main__":
    main()
