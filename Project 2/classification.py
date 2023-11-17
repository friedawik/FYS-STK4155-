import os
import pathlib
import numpy as np
from random import random, seed
from IPython import embed
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import csv
from sklearn.datasets import load_breast_cancer
from functions import *
from ffnn import *

# This scripts performs a gris search for all model set-ups
# and writes the results to a scv file for further analysis

# Get data
seed(2023)
wisconsin = load_breast_cancer()
X = wisconsin.data
target = wisconsin.target
target = target.reshape(target.shape[0], 1)
X_train, X_val, t_train, t_val = train_test_split(X, target)

# Scale data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)

# Network architecture
input_nodes = X_train.shape[1]
output_nodes = 1

architecture_dict = {'architecture_1':(input_nodes,100,output_nodes),
              'architecture_2':(input_nodes,50,50,50,output_nodes),
              'architecture_3':(input_nodes,100,100,100,output_nodes)
              }


# gradient descent parameters
epochs = 100
batches = 10

# Perform grid search over lambda and eta
sns.set()
eta_vals=np.logspace(-10, 4, 5)
lmbd_vals=np.logspace(2, 4, 5)
train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

# Dictionary to be used in loop
hidden_func_dict = {'Sigmoid':sigmoid,
              'RELU':RELU,
              'Leaky RELU':LRELU
              }

output_func=sigmoid
output_func_name='Sigmoid'

# Dictionary to be used in loop
scheduler_dict = {
            'Constant': Constant, 
            'Adagrad': Adagrad,
            'RMS_prop': RMS_prop,
            'Adam': Adam, 
            'Momentum': Momentum, 
            'AdagradMomentum':AdagradMomentum 
            }
result_dict = {'Architecture':[], 'Hidden function':[], 'Scheduler':[], 'Eta':[], 'Lambda':[], 'Train accuracy':[], 'Test accuracy':[], 'Epochs':[], 'Batches':[]}

for architecture_name, architecture in architecture_dict.items():
    for hidden_func_name, hidden_func in hidden_func_dict.items():
        for scheduler_name, scheduler_func in scheduler_dict.items():
            results_dir = 'results/classification/'+architecture_name+'/'+hidden_func_name+'/'+scheduler_name+'/'
            results_dir_eta_lmda = results_dir+'eta_lmbda/'
            pathlib.Path(results_dir_eta_lmda).mkdir(parents=True,exist_ok=True)

            for i, eta in enumerate(eta_vals):
                for j, lmbd in enumerate(lmbd_vals):
                    neural_network = FFNN(architecture, hidden_func=hidden_func, output_func=output_func, cost_func=CostLogReg, seed=2023)
                    neural_network.reset_weights() # reset weights such that previous runs or reruns don't affect the weights 
                    # Match string with function that contains right parameters
                    match scheduler_func.__name__:
                        case 'Constant': scheduler = scheduler_func(eta=eta)
                        case 'Adagrad': scheduler = scheduler_func(eta=eta)
                        case 'RMS_prop': scheduler = scheduler_func(eta=eta, rho=0.9)
                        case 'Adam': scheduler = scheduler_func(eta=eta, rho=0.9, rho2=0.999)
                        case 'Momentum': scheduler = scheduler_func(eta=eta, momentum = 0.3)
                        case 'AdagradMomentum': scheduler = scheduler_func(eta=eta, momentum = 0.3)

                    try:
                        scores = neural_network.fit(X_train, t_train, scheduler, epochs=epochs, batches=batches, X_val=X_val, t_val=t_val, lam=lmbd)
                        train_accuracy[i][j] = scores['train_accs'][-1]
                        test_accuracy[i][j] = scores['val_accs'][-1]
                    except:
                        train_accuracy[i][j] = np.nan
                        test_accuracy[i][j] = np.nan
                    result_dict['Architecture'].append(architecture_name)
                    result_dict['Hidden function'].append(hidden_func_name)
                    result_dict['Scheduler'].append(scheduler_name)
                    result_dict['Eta'].append(eta)
                    result_dict['Lambda'].append(lmbd)
                    result_dict['Epochs'].append(epochs)
                    result_dict['Batches'].append(batches)
                    result_dict['Train accuracy'].append(train_accuracy[i][j])
                    result_dict['Test accuracy'].append(test_accuracy[i][j])

                    # Plot convergence
                    fig1 = plt.figure()
                    fig_title = ( 'Hidden: ' + hidden_func_name+' | '+scheduler_name+' \n Batches: ' +str(batches) +' | Eta: ' + "{:.1E}".format(eta) + ' | Lambda: ' + "{:.1E}".format(lmbd) )
                    
                    fig_name = (results_dir_eta_lmda+'eta'+str(i)+'_lmbda'+str(j))
                    plt.plot(np.arange(epochs), scores['val_accs'] )
                    plt.xlabel('Epochs'), plt.ylabel('Test accuracy'), plt.title(fig_title)
                    plt.savefig(fig_name)
                    plt.close()
            
            # Plot train accuracy heatmap and save figure
            fig1, ax = plt.subplots(figsize = (10, 10))
            sns.heatmap(train_accuracy, annot=False, ax=ax, cmap="viridis",vmin=0, vmax=1)
            ax.set_ylabel("$\eta$")
            ax.set_xlabel("$\lambda$")
            ax.set_title(f"Training accuracy {architecture_name}\n Hidden: {hidden_func_name} | Out: {output_func_name} | {scheduler_func.__name__} | {epochs} Epochs | {batches} Batches ")
            fig_name = results_dir + f'sns_train_{scheduler_func.__name__}_{epochs}_epochs_{hidden_func_name}_{output_func_name}'
            plt.savefig(fig_name)
            plt.close()

            # Plot test accuracy heatmap and save figure
            fig1, ax = plt.subplots(figsize = (10, 10))
            sns.heatmap(test_accuracy, annot=False, ax=ax, cmap="viridis",vmin=0, vmax=1)
            ax.set_ylabel("$\eta$")
            ax.set_xlabel("$\lambda$")
            ax.set_title(f"Test accuracy {architecture_name}\n Hidden: {hidden_func_name} | Out: {output_func_name} | {scheduler_func.__name__} | {epochs} Epochs | {batches} Batches ")
            fig_name = results_dir + f'sns_test_{scheduler_func.__name__}_{epochs}_epochs_{hidden_func_name}_{output_func_name}'
            plt.savefig(fig_name)
            plt.close()
    

# make csv file for results

result_pd = pd.DataFrame.from_dict(result_dict)
result_pd.to_csv('temp_results_pd')
