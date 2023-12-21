from sklearn.model_selection import ParameterGrid
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from pytorch_cnn import *
import pandas as pd
import random
import time

# Seed for reproducibility
torch.manual_seed(0)
random.seed(0)
# Set device (CUDA if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Batch size
batch_size = 200
epochs = 10

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


#model = FFNN()

# Define the hyperparameters you want to tune and their search space
param_grid = {
    'lr': np.logspace(-4, 1, 8),
    'weight_decay': np.logspace(-4, 1, 8)
}
# Generate all combinations of hyperparameters
grid = list(ParameterGrid(param_grid))
# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
# Initialise dict for saving hyperparameters
hyperparameters_dict = {'Test accuracy':[], 'Train accuracy':[],'Learning rate': [], 'Weight decay':[], 'Batch size':[], 'Epoch':[]}

for i,params in enumerate(grid):
    t = time.time()  
    print(f'Round {i+1} of {len(grid)} with parameters {params}')

    try:
        # Define the model architecture
        model = CNN(kernel_size = 3, 
                    pooling_size = 2, 
                    padding_size = 1,
                    stride = 1,
                    image_dim = (batch_size, 3,32,32)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

        for ep in range(epochs):
            t2 = time.time() 
            print(f"Epoch {ep+1}\n-------------------------------")
            train_acc = train_loop(train_loader, model, loss_fn, optimizer)
            test_acc = test_loop(test_loader, model, loss_fn)

            elapsed = time.time() - t2
            print(f'Time used for caculation epoch {ep+1}: {(elapsed):>0.1f}')

            t3 = time.time() 
            hyperparameters_dict['Test accuracy'].append(test_acc)
            hyperparameters_dict['Train accuracy'].append(train_acc)
            hyperparameters_dict['Learning rate'].append(params['lr'])
            hyperparameters_dict['Batch size'].append(batch_size)
            hyperparameters_dict['Weight decay'].append(params['weight_decay'])
            hyperparameters_dict['Epoch'].append(ep+1)
            elapsed = time.time() - t3
            print(f'Time used for saving epoch {ep+1}: {(elapsed):>0.1f}')

    except:
        hyperparameters_dict['Test accuracy'].append(np.nan)
        hyperparameters_dict['Train accuracy'].append(np.nan)
        hyperparameters_dict['Epoch'].append(np.nan)
        hyperparameters_dict['Learning rate'].append(params['lr'])
        hyperparameters_dict['Batch size'].append(batch_size)
        hyperparameters_dict['Weight decay'].append(params['weight_decay'])


    print('results')
    results = pd.DataFrame.from_dict(hyperparameters_dict)  
    results.to_csv('results_cnn.csv')  
    #torch.save(model.state_dict(), f"model_k{params['kernel_size']}_po{params['pooling_size']}_pa{params['padding_size']}_s{params['stride']}")

#torch.save(model.state_dict(), "model_cifar_ffnn.pth")


