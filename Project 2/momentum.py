import pandas as pd
from IPython import embed
import numpy as np
from ffnn import *
from functions import *
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import seaborn as sns
from sklearn.datasets import load_breast_cancer

# script to update momentum and adagrad w/momentum with best momentum values

# get data
seed(42)
wisconsin = load_breast_cancer()
X = wisconsin.data
target = wisconsin.target
target = target.reshape(target.shape[0], 1)
X_train, X_val, t_train, t_val = train_test_split(X, target, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)

# Network architecture
input_nodes = X_train.shape[1]
output_nodes = 1

## Read the saved results table
results = pd.read_csv('res_arch1_2_3_all.csv')
#results[results['Architecture'] != 'architecture_3'] # used for rerun of only arch_3

# Select the best runs for momentum
res=results.sort_values(['Test accuracy'],ascending=False).groupby(['Architecture', 'Hidden function', 'Scheduler']).head(1)
best_mom = res[res['Scheduler']=='Momentum']
index = best_mom.index

# Select the best runs for AdagradMomentum
res=results.sort_values(['Test accuracy'],ascending=False).groupby(['Architecture', 'Hidden function', 'Scheduler']).head(1)
best_ada_mom = res[res['Scheduler']=='AdagradMomentum']
index_ada_mom = best_ada_mom.index

# Define dimensions
architecture_dict = {'architecture_1':(input_nodes,100,output_nodes),
              'architecture_2':(input_nodes,50,50,50,output_nodes),
              'architecture_3':(input_nodes,101,101,101,output_nodes)
              }
# Define hidden functions
hidden_func_dict = {'Sigmoid':sigmoid,
              'RELU':RELU,
              'Leaky RELU':LRELU
              }
# Model set-up
epochs = 50
batches = 5
momentum = np.linspace(0.1,1,10)
mom_array = np.zeros(len(momentum))
mom_tried = 0.3 # This was the setting when running through all models
mom_update = 0 # initialise updated momentum
# add column momentum
results['Momentum']=''
# Run models with diff momentum and update results
for ind in index:
    test_acc = results['Test accuracy'][ind]
    for mom in momentum:
        results.loc[ind, 'Momentum'] = mom_tried 
        architecture = architecture_dict[results['Architecture'][ind]]
        hidden_func = hidden_func_dict[results['Hidden function'][ind]]
        lmbd = results['Lambda'][ind]
        scheduler = Momentum(eta=results['Eta'][ind], momentum=mom)
        neural_network = FFNN(dimensions=architecture, 
                              hidden_func=hidden_func, 
                              output_func=sigmoid, 
                              cost_func=CostLogReg, 
                              seed=2023)
        neural_network.reset_weights()
        try:
            scores = neural_network.fit(X_train, t_train, scheduler, 
                                        epochs=epochs, batches=batches, 
                                        X_val=X_val, t_val=t_val, lam=lmbd)
            new_acc = scores['val_accs'][-1]
            if  new_acc > test_acc:
                test_acc = new_acc
                mom_update = mom
                
        except:
            continue
    results.loc[ind, 'Test accuracy'] = test_acc
    results.loc[ind, 'Momentum'] = mom_update

# same for Adagrad with momentum
for ind in index_ada_mom:
    test_acc = results['Test accuracy'][ind]
    for mom in momentum:
        results.loc[ind, 'Momentum'] = mom_tried 
        architecture = architecture_dict[results['Architecture'][ind]]
        hidden_func = hidden_func_dict[results['Hidden function'][ind]]
        lmbd = results['Lambda'][ind]
        scheduler = AdagradMomentum(eta=results['Eta'][ind], momentum=mom)
        neural_network = FFNN(dimensions=architecture, 
                              hidden_func=hidden_func, 
                              output_func=sigmoid, 
                              cost_func=CostLogReg, 
                              seed=2023)
        neural_network.reset_weights()
        try:
            scores = neural_network.fit(X_train, t_train, scheduler, 
                                        epochs=epochs, batches=batches, 
                                        X_val=X_val, t_val=t_val, lam=lmbd)
            new_acc = scores['val_accs'][-1]
            if  new_acc > test_acc:
                test_acc = new_acc
                mom_update = mom
                
        except:
            continue
    results.loc[ind, 'Test accuracy'] = test_acc
    results.loc[ind, 'Momentum'] = mom_update

# Make new csv with updated momentum
results.to_csv('result_updated_momentum_seed42.csv')

