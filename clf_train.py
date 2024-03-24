# Import necessary libraries
import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchvision import transforms

import argparse
from PIL import Image
import numpy as np
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix

# Include the GroceryDataModule and LitModel class definitions from your code snippet here

class GroceryStoreDataset(Dataset):

    # directory structure:
    # - root/[train/test/val]/[vegetable/fruit/packages]/[vegetables/fruit/packages]_class/[vegetables/fruit/packages]_subclass/[vegetables/fruit/packages]_image.jpg
    # - root/classes.csv
    # - root/train.txt
    # - root/test.txt
    # - root/val.txt
    def __init__(self, split='train', transform=None):
        super(GroceryStoreDataset, self).__init__()
        self.root = "/work/cvcs_2023_group23/GroceryStoreDataset/dataset/"
        self.split = split
        self.transform = transform
        self.class_to_idx = {}
        self.idx_to_class = {}
        self._descriptions = {}

        classes_file = os.path.join(self.root, "classes.csv")

        self.classes = {'81': 'background'}
        with open(classes_file, "r") as f:

            lines = f.readlines()

            for line in lines[1:]:
                class_name, class_id, coarse_class_name, coarse_class_id, iconic_image_path, prod_description = line.strip().split(
                    ",")
                self.classes[class_id] = class_name
                self.class_to_idx[class_name] = coarse_class_id
                self.idx_to_class[class_id] = class_name
                self._descriptions[class_name] = prod_description

        self.samples = []
        split_file = os.path.join(self.root, self.split + ".txt")
        # print(self.classes)
        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                img_path, class_id, coarse_class_id = line.split(",")
                class_name = self.classes[class_id.strip()]
                self.samples.append(
                    (os.path.join(self.root, img_path), int(self.class_to_idx[class_name])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        # print(img.shape, label)
        return img, label

    def description(self, class_name):
        return self._descriptions[class_name]

class GroceryDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, data_dir="/work/cvcs_2023_group23/GroceryStoreDataset/dataset/"):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            # Consider adding normalization if necessary
        ])
        self.groceery_train, self.grocery_test, self.grocery_val = None, None, None

    def setup(self, stage=None):
        # Assign training/validation datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.grocery_train = GroceryStoreDataset(split='train', transform=self.transform)
            self.grocery_val = GroceryStoreDataset(split='val', transform=self.transform)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.grocery_test = GroceryStoreDataset(split='test', transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.grocery_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.grocery_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.grocery_test, batch_size=self.batch_size)

class LitModel(pl.LightningModule):
    def __init__(self, num_classes=82, learning_rate=1e-3):
        super().__init__()
        self.model = models.densenet121(pretrained=True)
        self.model.classifier = nn.Linear(1024, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adamax(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / labels.size(0)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('test_loss', loss)

        # Compute top-k accuracies
        top1_acc = self.compute_topk_accuracy(outputs, labels, k=1)
        top3_acc = self.compute_topk_accuracy(outputs, labels, k=3)
        top5_acc = self.compute_topk_accuracy(outputs, labels, k=5)
        self.log_dict({'test_top1_acc': top1_acc, 'test_top3_acc': top3_acc, 'test_top5_acc': top5_acc})

        # Add preds and labels to output
        _, preds = torch.max(outputs, dim=1)
        return {'test_loss': loss, 'preds': preds, 'labels': labels, 'test_top1_acc': top1_acc, 'test_top3_acc': top3_acc, 'test_top5_acc': top5_acc}

    def compute_topk_accuracy(self, outputs, labels, k=1):
        _, top_k_predictions = outputs.topk(k, 1, True, True)
        top_k_correct = top_k_predictions.eq(labels.view(-1, 1).expand_as(top_k_predictions))
        top_k_correct_sum = top_k_correct.view(-1).float().sum(0)
        return top_k_correct_sum.mul_(100.0 / labels.size(0))

print("Starting clf_train.py \n")

# Set constants for the script
EPOCHS = 40
BATCH_SIZE = 64
NUM_CLASSES = 82  # Update this based on your dataset
DATA_DIR = "/work/cvcs_2023_group23/GroceryStoreDataset/dataset/"  # Update this path
CHECKPOINT_DIR = "checkpoints/clf_densetnet121/240324/40Epochs/"  # Update this path

# Additional import for argument parsing
import argparse

# Parse command line arguments for an optional checkpoint path
parser = argparse.ArgumentParser(description="Train the model with PyTorch Lightning.")
parser.add_argument("--checkpoint_path", type=str, default="", help="Path to a checkpoint to load and continue training.")
args = parser.parse_args()

# Initialize the model
model = LitModel(num_classes=NUM_CLASSES)
# Check if a checkpoint path was provided and load it
if args.checkpoint_path:
    model = LitModel.load_from_checkpoint(args.checkpoint_path)

# Initialize the data module
data_module = GroceryDataModule(batch_size=BATCH_SIZE, data_dir=DATA_DIR)

# Setup logger and checkpoint callback
logger = TensorBoardLogger('logs/clf_densenet121', name='tb_logs')
checkpoint_callback = ModelCheckpoint(
    dirpath=CHECKPOINT_DIR,
    filename='last.ckpt',
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    every_n_epochs=1,
    save_last=True 
)

# Initialize the trainer
trainer = Trainer(
    max_epochs=EPOCHS,
    logger=logger,
    callbacks=[RichProgressBar(), checkpoint_callback, EarlyStopping(monitor='val_loss')],
    accelerator='auto',
    devices='auto',
    resume_from_checkpoint=args.checkpoint_path
)

# Train the model
trainer.fit(model, datamodule=data_module)