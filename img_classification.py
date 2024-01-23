import math
import os.path
import numpy as np
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.models import GoogLeNet_Weights

from datasets FreiburgDataset, GroceryStoreDataset
from tqdm import tqdm


trainset = FreiburgDataset()
valset = FreiburgDataset()

num_classes = 25

trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
valloader = DataLoader(valset, batch_size=32, shuffle=True, num_workers=2)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on {device}...')
model = torchvision.models.densenet121(pretrained=True).to(device)

# model = torchvision.models.resnet18().to(device)
model.classifier = nn.Linear(
    1024,  num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adamax(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=3, verbose=True)

if os.path.exists('classifier.pth'):
    model.load_state_dict(torch.load('classifier.pth'))
epochs = 10

print('''
#################################################################
#                                                               #
#                       Training Started                        #
#                                                               #                 
#################################################################
''')
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
   

print(''''
#################################################################
#                                                               #
#                       Training Completed                      #
#                                                               #                 
#################################################################
''')

torch.save(model.state_dict(), 'classifier.pth')

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
plt.savefig('classification_results.png')
