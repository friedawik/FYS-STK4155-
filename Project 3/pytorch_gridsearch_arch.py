import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from IPython import embed
from sklearn.model_selection import ParameterGrid
from pytorch_cnn import *
import numpy as np
import pandas as pd
import random
import time

# Seed for reproducibility
torch.manual_seed(0)
random.seed(0)

# Set device (CUDA if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epochs = 10
batch_size = 200
learning_rate = 0.002683
weight_decay = 0.000518

# Get and transform dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Dictionary for saving results
architecture_dict = {'Test accuracy':[],'Train accuracy':[], 'Epoch':[], 'Kernel size':[], 'Padding size':[],'Pooling size':[],'Stride':[]}
# Initialize loss function

loss_fn = nn.CrossEntropyLoss()

param_grid = {
    'kernel_size': [3, 4, 5],  # Size of kernel
    'pooling_size': [2, 3, 4],  # Size of pooling
    'padding_size': [1],  # Size of padding
    'stride': [1, 2, 3],  # stride
}

# Generate all combinations of hyperparameters
grid = list(ParameterGrid(param_grid))

for i, params in enumerate(grid):
    t = time.time()  
    print(f'Round {i+1} of {len(grid)} with parameters {params}')

    try: 
        model = CNN(kernel_size = params['kernel_size'], 
                pooling_size = params['pooling_size'], 
                padding_size = params['padding_size'],
                stride = params['stride'],
                image_dim = (batch_size, 3,32,32)).to(device)
    
    
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        for t in range(num_epochs):
            t2 = time.time() 
            print(f"Epoch {t+1}\n-------------------------------")
            train_acc = train_loop(train_loader, model, loss_fn, optimizer)
            test_acc = test_loop(test_loader, model, loss_fn)
            elapsed = time.time() - t2
            print(f'Time used for caculation epoch {t+1}: {(elapsed):>0.1f}')
            architecture_dict['Test accuracy'].append(test_acc)
            architecture_dict['Train accuracy'].append(train_acc)
            architecture_dict['Epoch'].append(t)
            architecture_dict['Pooling size'].append(params['pooling_size'])
            architecture_dict['Padding size'].append(params['padding_size'])
            architecture_dict['Kernel size'].append(params['kernel_size'])
            architecture_dict['Stride'].append(params['stride'])
    except:
        architecture_dict['Test accuracy'].append(np.nan)
        architecture_dict['Train accuracy'].append(np.nan)
        architecture_dict['Epoch'].append(np.nan)
        architecture_dict['Pooling size'].append(params['pooling_size'])
        architecture_dict['Padding size'].append(params['padding_size'])
        architecture_dict['Kernel size'].append(params['kernel_size'])
        architecture_dict['Stride'].append(params['stride'])


    results = pd.DataFrame.from_dict(architecture_dict)  
    results.to_csv('results_cnn_arch.csv') 
    
    torch.save(model.state_dict(), f"cnn_arch/model_k{params['kernel_size']}_po{params['pooling_size']}_pa{params['padding_size']}_s{params['stride']}")


