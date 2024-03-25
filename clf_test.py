# Core libraries
import os
import argparse
from PIL import Image
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Torchvision
from torchvision import transforms, models
from torchvision.utils import save_image

# Visualization & Metrics
import numpy as np
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
    def __init__(self, split='test', transform=None):
        super(GroceryStoreDataset, self).__init__()
        self.root = "/work/cvcs_2023_group23/GroceryStoreDataset/dataset/"
        self.split = split
        self.transform = transform
        self.class_to_idx = {}
        self.idx_to_class = {}
        self._descriptions = {}

        classes_file = os.path.join(self.root, "classes.csv")

        self.classes = {'42': 'background'}
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
    def __init__(self, batch_size, data_dir):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            # Consider adding normalization if necessary
        ])
        self.grocery_train, self.grocery_test, self.grocery_val = None, None, None

    def setup(self, stage=None):
        # Assign training/validation datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.grocery_train = GroceryStoreDataset(split='train', transform=self.transform)
            self.class_weights = self.calculate_class_weights()
            self.grocery_val = GroceryStoreDataset(split='val', transform=self.transform)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.grocery_test = GroceryStoreDataset(split='test', transform=self.transform)
    
    def calculate_class_weights(self):
        # Count the number of instances of each class
        class_counts = torch.zeros(43)
        for _, label in self.grocery_train:
            class_counts[label] += 1
        # Inverse of counts to get weights
        class_weights = 1. / class_counts
        return class_weights

    def train_dataloader(self):
            sample_weights = [self.class_weights[label] for _, label in self.grocery_train]
            weighted_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            return DataLoader(self.grocery_train, batch_size=self.batch_size, sampler=weighted_sampler)
    
    def val_dataloader(self):
        return DataLoader(self.grocery_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.grocery_test, batch_size=self.batch_size)

class LitModel(pl.LightningModule):
    def __init__(self, num_classes=43, learning_rate=1e-3):
        super().__init__()
        # for param in self.model.parameters():
        #     param.requires_grad = False
        
        self.model = models.densenet121(pretrained=True)
        num_ftrs = self.model.classifier.in_features  # Get the number of features of the last layer
        self.model.classifier = nn.Linear(num_ftrs, num_classes)  # Update classifier
                
        # Make sure the classifier parameters are set to require gradients
        # for param in self.model.classifier.parameters():
        #     param.requires_grad = True
        
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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Perform inference on test dataset.")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save inference outputs.")
    return parser.parse_args()

def save_results(predictions, labels, output_dir):
    results = {"predictions": predictions, "labels": labels}
    with open(os.path.join(output_dir, "inference_results.json"), "w") as f:
        json.dump(results, f)
    print("Inference results saved.")

def main():
    args = parse_arguments()
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the model from the checkpoint
    model = LitModel.load_from_checkpoint(checkpoint_path=args.model_checkpoint).eval().to(device)
    
    # Define test dataset and loader
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
    ])
    test_dataset = GroceryStoreDataset(split='test', transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    
    # Prepare lists to store predictions and their correctness
    all_preds = []
    all_labels = []
    correctness_list = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Convert logits to probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)
            # Get the top prediction for each input
            _, top_preds = torch.max(probs, dim=1)
            
            for j, image in enumerate(images):
                # Find the predicted class name and the correctness for each image
                pred_class_id = top_preds[j].item()
                true_class_id = labels[j].item()
                pred_class_name = test_dataset.idx_to_class[str(pred_class_id)]
                is_correct = pred_class_id == true_class_id
                
                # Append predictions and correctness to lists
                all_preds.append(pred_class_id)
                all_labels.append(true_class_id)
                correctness_list.append(is_correct)
                
                # Save the image with the inferred class and correctness in the filename
                correctness_label = "correct" if is_correct else "incorrect"
                save_path = os.path.join(args.output_dir, f"inferred_images/{i}_{j}_class_{pred_class_name}_{correctness_label}.jpg")
                save_image(image.cpu(), save_path)
                
    # After processing all images, save the results to a JSON file
    results = {
        "predictions": all_preds,
        "labels": all_labels,
        "correctness": correctness_list
    }
    results_path = os.path.join(args.output_dir, "inference_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f)
    print("Inference images and results saved.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Starting clf_test.py \n")
    main()