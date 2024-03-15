import math
import os.path
import numpy as np
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, WeightedRandomSampler
# from torchvision.models import GoogLeNet_Weights
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor

from datasets import FreiburgDataset, TEST_TRANSFORM, collate_fn
from tqdm import tqdm

num_classes = 25

# Evaluation part

testset = FreiburgDataset(split='full_testing', transform=TEST_TRANSFORM)
testloader = DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on {device}...')
model = torchvision.models.densenet121(pretrained=True).to(device)

# model = torchvision.models.resnet50().to(device)
model.classifier = nn.Sequential(
    nn.Linear(
    1024,  256).to(device),
    nn.Linear(256, num_classes).to(device)
    )
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adamax(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=3, verbose=True)

if os.path.exists('freiburg/single/classifier.pth'):
    model.load_state_dict(torch.load('freiburg/single/classifier.pth'))

accuracy = 0
total = 0
correct = 0
with torch.no_grad():
        for idx, data in enumerate(testloader):
            images, test_labels = data
            images = images.to(device)
            test_labels = test_labels.to(device)

            test_outputs = model(images)
            _, predicted = torch.max(test_outputs.data, 1)
            

            total += test_labels.size(0)
            correct += (predicted == test_labels).sum().item()
        accuracy = 100 * correct / total

print(f"Evaluation ended.\n\n Accuracy: {accuracy}%")