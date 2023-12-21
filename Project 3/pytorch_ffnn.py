import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import ParameterGrid
import numpy as np
import pandas as pd
import time
from IPython import embed
import random


# Seed for reproducibility
torch.manual_seed(0)
random.seed(0)

# possibility of using graphic card
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 10
batch_size = 100
learning_rate = 0.001

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the images
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# FFNN model
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Model initialization
input_size = 32 * 32 * 3  # CIFAR-10 image shape: 32x32x3
hidden_size = 100
num_classes = 10
model = FFNN(input_size, hidden_size, num_classes).to(device)

param_grid = {
    'lr': np.logspace(-4, 1, 8),
    'weight_decay': np.logspace(-4, 1, 8)
}

grid = list(ParameterGrid(param_grid))
results_dict = {'Test accuracy':[], 'Train accuracy':[], 'Train loss':[],'Learning rate': [], 'Weight decay':[], 'Epoch':[]}

for i,params in enumerate(grid):
    t = time.time()  
    print(f'Round {i+1} of {len(grid)} with parameters {params}')

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    # Training loop
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            train_loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            
            # Backward and optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            train_accuracy = (correct / total) * 100
            
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {train_loss.item():.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        test_accuracy = (correct / total) * 100
        print(f'Test Accuracy: {test_accuracy}%')
        results_dict['Test accuracy'].append(test_accuracy)
        results_dict['Train accuracy'].append(train_accuracy)
        results_dict['Train loss'].append(train_loss.item())
        results_dict['Learning rate'].append(params['lr'])
        results_dict['Weight decay'].append(params['weight_decay'])
        results_dict['Epoch'].append(epoch+1)
        elapsed = time.time() - t
        print(f'Time used for epoch {epoch+1}: {(elapsed):>0.1f}')

    results = pd.DataFrame.from_dict(results_dict)  
    results.to_csv('results_ffnn.csv') 

