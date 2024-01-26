import math
import os.path
import numpy as np
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
# from torchvision.models import GoogLeNet_Weights

from datasets import FreiburgDataset, GroceryStoreDataset
from tqdm import tqdm
import random

# Define the number of models for the ensemble
num_models = 5
models = {}
dataloaders = {}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_classes = 25
epochs = 5
criterion = nn.CrossEntropyLoss()

# Split dataset into subsets and initialize models
for i in range(num_models):
    subset = FreiburgDataset(split='train', index=i)
    dataloaders[i] = DataLoader(subset, batch_size=32, shuffle=True, num_workers=2)
    models[i] = torchvision.models.densenet121(pretrained=True).to(device)
    models[i].classifier = nn.Linear(1024, num_classes).to(device)

# Train each model on its subset
for model_index, model in models.items():
    print(f"Training Model {model_index + 1}/{num_models}")
    optimizer = optim.Adamax(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=3, verbose=True)

    # Check if a saved model exists
    if os.path.exists(f'freiburg/classifier_{model_index}.pth'):
        model.load_state_dict(torch.load(f'freiburg/classifier_{model_index}.pth'))

    trainloader = dataloaders[model_index]

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(trainloader, desc=f'Epoch {epoch + 1}/{epochs} (Model {model_index + 1})', unit='batch')

        for i, data in enumerate(pbar):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (i + 1)})
            model.eval()
            valset = FreiburgDataset(split='val', index=int(model_index % 2))
            val_loader = DataLoader(valset, batch_size=16, shuffle=True, num_workers=2)
            
            total_val_loss = 0
            for eval_idx, eval_data in enumerate(val_loader):
                
                inputs, labels = eval_data
                inputs, labels = inputs.to(device), labels.to(device)
                out = model(inputs)
                
                val_loss = criterion(out, labels)
                
                total_val_loss += loss.item()
                
        # Scheduler step
        scheduler.step(total_val_loss)

        # Save the model after each epoch
        torch.save(model.state_dict(), f'freiburg/classifier_{model_index}.pth')

    print(f"Training for Model {model_index + 1} completed")


# Combine predictions during evaluation
def ensemble_predict(models, dataloader):
    total_preds = None
    for model in models.values():
        model.eval()
        with torch.no_grad():
            for data in dataloader:
                inputs, _ = data
                inputs = inputs.to(device)
                outputs = model(inputs)
                if total_preds is None:
                    total_preds = outputs
                else:
                    total_preds += outputs
    return total_preds / len(models)

# Evaluation
testset = FreiburgDataset(split="test")
testloader = DataLoader(testset, batch_size=16, shuffle=True, num_workers=2)
accuracy = 0
total = 0
correct = 0
for idx, data in enumerate(testloader):
    images, test_labels = data
    images = images.to(device)
    test_labels = test_labels.to(device)
    
    ensemble_outputs = ensemble_predict(models, [(images, test_labels)])
    _, predicted = torch.max(ensemble_outputs.data, 1)
    
    total += test_labels.size(0)
    correct += (predicted == test_labels).sum().item()
accuracy = 100 * correct / total

print(f"Evaluation ended.\n\n Accuracy: {accuracy}")
