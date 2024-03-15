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

from datasets import FreiburgDataset, TRAIN_TRANSFORM, TEST_TRANSFORM, VAL_TRANSFORM, collate_fn
from tqdm import tqdm

num_classes = 25

# create a custom dataset class for each dataset
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# trainset = FreiburgDataset(split='full_training')
# valset = FreiburgDataset(split='val')
# trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
# valloader = DataLoader(valset, batch_size=32, shuffle=True, num_workers=2)

# Create the dataset for the training split
trainset = FreiburgDataset(split='full_training', transform=TRAIN_TRANSFORM)
valset = FreiburgDataset(split='val', transform=VAL_TRANSFORM)
#testset = FreiburgDataset(split='test', transform=TEST_TRANSFORM)

# Handling class imbalance
class_sample_count = np.array([len(np.where(trainset.targets == t)[0]) for t in np.unique(trainset.targets)])
weight = 1. / (class_sample_count + np.finfo(float).eps)  # Adding a small epsilon value
unique_classes = np.unique(trainset.targets)
class_weights = {cls: 1. / (np.sum(trainset.targets == cls) + np.finfo(float).eps) for cls in unique_classes}
samples_weight = np.array([class_weights[t] for t in trainset.targets])
samples_weight = torch.from_numpy(samples_weight)
samples_weight = samples_weight.double()
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

# Create a DataLoader
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
valloader = DataLoader(valset, batch_size=32, shuffle=True, num_workers=2)
# trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2, collate_fn=collate_fn)
# valloader = DataLoader(valset, batch_size=32, shuffle=True, num_workers=2, collate_fn=collate_fn)
#testloader = DataLoader(testset, batch_size=32, num_workers=0, collate_fn=collate_fn)

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
epochs = 10

losses = []
val_losses = []
acc = 0
# training loop
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    pbar = tqdm(trainloader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch')

    for i, data in enumerate(pbar):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        pbar.set_postfix({'loss': running_loss / (i + 1), 'val_accuracy': acc})
    if i == len(trainloader) - 1:
        if losses:
            if losses[-1] < running_loss:
                print("Possible overfit...")
        losses.append(running_loss/(i+1))
        
    model.eval()
    correct = 0
    val_loss = 0
    total = 0
    with torch.no_grad():
        for idx, data in enumerate(valloader):
            images, val_labels = data
            images = images.to(device)
            val_labels = val_labels.to(device)

            val_outputs = model(images)
            _, predicted = torch.max(val_outputs.data, 1)
            val_loss += criterion(val_outputs, val_labels).item()

            total += val_labels.size(0)
            correct += (predicted == val_labels).sum().item()
        acc = 100 * correct / total
        val_loss /= idx+1
        val_losses.append(val_loss)
    
    model.train()
    scheduler.step(val_loss)
    # if epoch % 2 == 0:
    #     torch.save(model.state_dict(), 'freiburg/single/classifier.pth')
    torch.save(model.state_dict(), 'freiburg/single/classifier.pth')
    print('model saved')

print('''
#################################################################
#                                                               #
#                       Training Completed                      #
#                                                               #                 
#################################################################
''')

torch.save(model.state_dict(), 'freiburg/single/classifier.pth')

print('''
#################################################################
#                                                               #
#                       Model Saved                             #
#                                                               #                 
#################################################################
''')
x = np.linspace(0, epochs, epochs)
fig, ax = plt.subplots()
ax.plot(x, losses, label="training loss")
ax.plot(x, val_losses, label="validation loss")
ax.legend()
plt.savefig('freiburg/single/classification_results.png')

## Evaluation part

## testset = FreiburgDataset(split='full_testing')
## testloader = DataLoader(testset, batch_size=16, shuffle=True, num_workers=2)

# testset = FreiburgDataset(split='full_testing', transform=TEST_TRANSFORM)
# testloader = DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)

# accuracy = 0
# total = 0
# correct = 0
# with torch.no_grad():
#         for idx, data in enumerate(testloader):
#             images, test_labels = data
#             images = images.to(device)
#             test_labels = test_labels.to(device)

#             test_outputs = model(images)
#             _, predicted = torch.max(test_outputs.data, 1)
            

#             total += test_labels.size(0)
#             correct += (predicted == test_labels).sum().item()
#         accuracy = 100 * correct / total

# print(f"Evaluation ended.\n\n Accuracy: {accuracy}%")
