import os
import argparse
import logging
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from datasets import SKUDataset, TRAIN_TRANSFORM, VAL_TRANSFORM, custom_collate_fn
import yaml
from torch.optim import SGD
#from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def setup_logging(log_dir, verbose):
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename=os.path.join(log_dir, 'train.log'), level=logging.INFO if verbose else logging.WARNING, format=log_format)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(console_handler)

def train(args, config, tensorboard_writer):
    # Data loading setup
    train_dataset = SKUDataset(split='train', transform=TRAIN_TRANSFORM)
    val_dataset = SKUDataset(split='val', transform=VAL_TRANSFORM)
    train_dataloader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=2, collate_fn=custom_collate_fn, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['training']['batch_size']//2, shuffle=True, num_workers=2, collate_fn=custom_collate_fn, pin_memory=True)

    # Model setup
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)  # Assuming 2 classes, change this to your number of classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Optimizer setup
    optimizer = SGD(model.parameters(), lr=config['training']['learning_rate'], weight_decay=0.01)

    # ReduceLROnPlateau scheduler decreases learning rate when a metric has stopped improving
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    
    start_epoch = 0
    num_epochs = config['training']['num_epochs']
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_progress = tqdm(train_dataloader, desc=f'Epoch [{epoch+1}/{num_epochs}]', leave=True)
        total_train_loss = 0.0
        for batch_idx, (images, x1, y1, x2, y2, class_id, image_width, image_height) in enumerate(train_progress):
            images = images.to(device)
            
            # Create a list of target dictionaries
            targets = []
            for i in range(len(images)):
                target = {}
                target['boxes'] = torch.stack((x1[i], y1[i], x2[i], y2[i]), dim=1)
                target["boxes"] = target["boxes"].to(device)
                target['labels'] = class_id[i].to(device)
                targets.append(target)

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass and optimization
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            total_train_loss += losses.item()
            train_loss = total_train_loss / (batch_idx + 1)
            tensorboard_writer.add_scalar('Loss/Train', train_loss, epoch * len(train_dataloader) + batch_idx)
            train_progress.set_postfix({'Loss': train_loss})

        """ # Validation
        model.eval()
        val_progress = tqdm(val_dataloader, desc=f'Validation [{epoch+1}/{num_epochs}]', leave=True)
        total_val_loss = 0.0
        for batch_idx, (images, x1, y1, x2, y2, class_id, image_width, image_height) in enumerate(val_progress):
            images = images.to(device)
            
            # Create a list of target dictionaries
            targets = []
            for i in range(len(images)):
                target = {}
                target['boxes'] = torch.stack((x1[i], y1[i], x2[i], y2[i]), dim=1)
                target["boxes"] = target["boxes"].to(device)
                target['labels'] = class_id[i].to(device)
                targets.append(target)

            with torch.no_grad():
                loss_list = model(images, targets)
                losses = sum(loss for loss in loss_list)
                total_val_loss += losses.item()
            
            val_loss = total_val_loss / (batch_idx + 1)
            tensorboard_writer.add_scalar('Loss/Validation', val_loss, epoch * len(val_dataloader) + batch_idx)
            val_progress.set_postfix({'Loss': val_loss})
 """
        # Adjust learning rate
        lr_scheduler.step(metrics=losses.item())

        logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {losses.item():.4f}')
        checkpoint_path = os.path.join("checkpoints","frcnn", 'checkpoint.pth')
        torch.save({
            'epoch': epoch,
            'batch_idx': len(train_dataloader) - 1,  # Last batch index
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'total_train_loss': total_train_loss
        }, checkpoint_path)

def main():
    parser = argparse.ArgumentParser(description="SKU Training Script")
    parser.add_argument("--config", required=True, help="Path to the YAML config file")
    parser.add_argument("--log_dir", required=True, help="Path to the log directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Load config file
    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    # Overwrite config with CLI arguments
    config['logging']['log_dir'] = args.log_dir
    config['training']['batch_size'] = args.batch_size
    config['training']['learning_rate'] = args.learning_rate
    config['training']['num_epochs'] = args.num_epochs

    # Setup logging
    setup_logging(config['logging']['log_dir'], args.verbose)

    # Tensorboard setup
    tensorboard_writer = SummaryWriter(config['logging']['log_dir'])

    # Training
    train(args, config, tensorboard_writer=tensorboard_writer)

if __name__ == '__main__':
    main()