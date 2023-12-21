import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from IPython import embed
from math import floor
import numpy as np
import time

# Set device (CUDA if available, otherwise CPU)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# CNN model
class CNN(nn.Module):
    def __init__(self, kernel_size, pooling_size, padding_size, stride, image_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=kernel_size, stride=stride, padding=padding_size)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=pooling_size)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride, padding=padding_size)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=pooling_size)

        # Calculate the input of the Linear layer   
        t = time.time()  
        conv1_out = self.get_output_shape(self.maxpool1, self.get_output_shape(self.conv1, image_dim))
        conv2_out = self.get_output_shape(self.maxpool2, self.get_output_shape(self.conv2, conv1_out)) 
        # multiply heigh, weight and channels to get flattened shape
        fc1_in = list(conv2_out)[1]*list(conv2_out)[2]*list(conv2_out)[3]
        self.fc = nn.Linear(fc1_in, 10)
        elapsed = time.time() - t
        #print(f'Time used: {(elapsed):>0.1f}')

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
    def get_output_shape(self, model, image_dim):
        output_shape = model(torch.rand(*(image_dim))).data.shape
        return output_shape


class FFNN(nn.Module):
    def __init__(self):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 1000)  # Input layer
        self.fc2 = nn.Linear(1000, 500)  # Hidden layer
        self.fc3 = nn.Linear(500, 10)  # Output layer

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input tensor
        x = torch.relu(self.fc1(x))  # Apply activation function to the hidden layer
        x = torch.relu(self.fc2(x))  # Apply activation function to the hidden layer
        x = self.fc3(x)
        return x
    
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    correct /= size
    train_accuracy = 100*correct
    return train_accuracy

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    test_accuracy = 100*correct
    print(f"Test Error: \n Accuracy: {(test_accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")   
    return test_accuracy



