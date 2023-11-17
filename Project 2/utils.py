import numpy as np
from random import random, seed
from IPython import embed
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.neural_network import MLPClassifier
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from sklearn import datasets
import seaborn as sns

# Make data
x = np.linspace(0,1,1001)
y = 1+x+x**2

# define simple polynomial design matrix
def simple_poly_design_matrix(x):  
        X = np.ones((len(x),3))
        #X = np.array(np.ones((len(x),3)), x[:,0],x[:,0]**2)
        X[:,1] = x[:,0]
        X[:,2] = x[:,0]**2
        return X


def franke_function(x,y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2) 
        #stdv = 1
        #noise = np.random.normal(0, stdv, x.shape[0])           # mean=0, standard deviation=1
        
        noise = 0.05 * np.random.randn(x.shape[0], y.shape[0])  # mean=0, standard deviation=1
        
        z = term1 + term2 + term3 + term4 + noise
        return z

class Feed_forward_nn:
        def __init__(self, X, Y, layer_dim, n_outputs, epochs, batch_size, eta, lmbd):
                seed(203)
                # All data
                self.X = X
                self.Y = Y
         
                # Length of input vectors and layers
                self.n_inputs = X.shape[0]
                self.n_features = X.shape[1]
        
                 # Model architecture
                self.n_layers = len(layer_dim)
                self.layer_dim = layer_dim
                self.n_outputs = n_outputs

                # Gradient descent parameters
                self.epochs = epochs
                self.eta = eta
                self.batch_size = batch_size
                self.iterations = self.n_inputs // self.batch_size
                self.lmbd = lmbd

                # Create starting weights and biases
                self.parameters = self.init_weights_and_biases()
                
        def init_weights_and_biases(self):
                param={}        # contains weight and biases
                # Initialize first hidden layers
                param[f'W{1}'] = np.random.randn(self.n_features, self.layer_dim[0])
                #param[f'b{1}'] = np.zeros((self.layer_dim[0], 1)) + 0.01
                param[f'b{1}'] = np.zeros(self.layer_dim[0]) + 0.01

                for l in range(1,self.n_layers):
                        param[f'W{l+1}'] = np.random.randn(self.layer_dim[l-1], self.layer_dim[l]) 
                        #param[f'b{l+1}'] = np.zeros((self.layer_dim[l], 1)) + 0.01
                        param[f'b{l+1}'] = np.zeros(self.layer_dim[l]) + 0.01
    
                param[f'W{self.n_layers+1}'] = np.random.randn(self.layer_dim[-1], self.n_outputs)
                param[f'b{self.n_layers+1}'] = np.zeros(self.n_outputs) + 0.01 
                return param


        def feed_forward(self):
                mem = {'A0':self.X_batch}  # dict mem contains all input z and output a of the layers
                for l in range(1,self.n_layers+2):       
                        z = mem[f'A{l-1}'] @ self.parameters[f'W{l}'] + self.parameters[f'b{l}']
                
                        if l == self.n_layers+1:                # output layer
                                mem[f'z{l}'] = z
                                #mem[f'A{l}'] = self.softmax(z)  # classification output
                                mem[f'A{l}'] = self.linear(z)               # regressuin output
                                #mem[f'A{l}'] = self.sigmoid(z) 
                        else:                                   # hidden layers
                                mem[f'z{l}'] = z
                                mem[f'A{l}'] = self.sigmoid(z)

                return mem
        
        def feed_forward_out(self, X):
                mem = {'A0':X}  # input layer 
                
                for l in range(1,self.n_layers+2):
                        z = mem[f'A{l-1}'] @ self.parameters[f'W{l}'] + self.parameters[f'b{l}']
                        
                        if l == self.n_layers+1:                # output layer
                                mem[f'z{l}'] = z
                                #mem[f'A{l}'] = self.softmax(z) # classification output
                                mem[f'A{l}'] = self.linear(z)   # regression output
                                #mem[f'A{l}'] = self.sigmoid(z)               
                        else:                                   # hidden layers
                                mem[f'z{l}'] = z
                                mem[f'A{l}'] = self.sigmoid(z)
                probabilities = mem[f'A{l}']
                
                #return np.argmax(probabilities, axis=1)   
                #return np.mean(probabilities, axis=1)    
        
                return probabilities

        
        def backpropagation(self, mem):        
                gradients = {}      
                a_L = mem[f'A{self.n_layers+1}']   
                #y_ = np.reshape(self.Y_batch, (len(self.Y_batch),-1))
                #error_prev = a_L*(1-a_L)*(a_L - self.Y_batch) # regression with sigmoid output
                
                error_prev = a_L  - self.Y_batch # classification or regression with linear out
            
               # error_prev = a_L  - y_ # classification
                for l in range(self.n_layers+1,0,-1):
                        dz = None
                        if l == self.n_layers+1:        # error output layer
                                dz = error_prev
                        else:      
                                dz =  error_prev @ self.parameters[f'W{l + 1}'].T * mem[f'A{l}']* (1-mem[f'A{l}']) # depends on activatoin function
                                #dz =  dA_prev @ self.parameters[f'W{l + 1}'].T * self.sigmoid_derivative(mem[f'z{l}']) # depends on activatoin function      
                        gradients[f'dW{l}'] = mem[f'A{l-1}'].T @ dz
                        gradients[f'db{l}'] = np.sum(dz, axis=0)
                        if self.lmbd > 0.0:
                                gradients[f'dW{l}'] += self.lmbd * self.parameters[f'W{l}']      
                        error_prev = dz
                return gradients

        def train(self):
        
                data_indices = np.arange(self.n_inputs)
                for i in range(self.epochs):
                        
                        for j in range(self.iterations):
                                # pick datapoints with replacement
                                chosen_datapoints = np.random.choice(data_indices, size=self.batch_size, replace=False)

                                # minibatch training data
                                self.X_batch = self.X[chosen_datapoints]
                                self.Y_batch = self.Y[chosen_datapoints] #[:, np.newaxis] 
         
                                mem = self.feed_forward()
                                gradients = self.backpropagation(mem)
                                
                                self.update_param(gradients)
                                

        def update_param(self, gradients):
                for l in range(1,self.n_layers+2): 
                             
                        self.parameters[f'W{l}'] -= self.eta * gradients[f'dW{l}']
                        self.parameters[f'b{l}'] -= self.eta * gradients[f'db{l}']
        
        def sigmoid(self, z):
                return 1 / (1 + np.exp(-z))
        
        def sigmoid_derivative(self, z):
               return self.sigmoid(z) * (1-self.sigmoid(z))
        
        def relu(self, z):
                return np.maximum(0, z)
        
        def linear(self,z):
                return z

        def softmax(self, z_o):
                exp_term = np.exp(z_o)                                # softmax equation
                probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
                return probabilities

        # one-hot in numpy
def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    
    return onehot_vector

def ols(X,y):
    beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y) # Find beta with matrix inversion (pseudoinverse)
    return beta 

def ridge(X, y, lambd):    
    size_train = X.shape[1] # Size for identity matrix   
    beta = np.linalg.pinv( X.T @ X + lambd * np.identity(size_train)) @ X.T @ y   
    return beta 