from utils import *
import numpy as np
import matplotlib.pyplot as plt
from random import random
from IPython import embed
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import linear_model
from scipy import linalg
from sklearn.utils import resample
import plotly.express as px
import pandas as pd
#from neural_network import Neural_network
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns 
from autograd import grad
import time
import seaborn as sns
import jax.numpy as jnp
from jax import grad, jit, vmap




# Make data
np.random.seed(2018)
n = 100             # datapoints

#x= np.linspace(0,100,n)         # too sorted data, needs small learning rate or no convergence
#x= np.sort(10*np.random.rand(n,1), axis=0)
x= np.sort(np.random.uniform(-10,10,(n,1)), axis=0)
y = (1+x+ x**2 + 4*np.random.randn(n,1))

# Design matrix of polynomial f(x)= 1+x+x**2
X = np.c_[np.ones((n,1)), x, x**2]
X = np.c_[ x, x**2]
# scaling
# scaler = StandardScaler()
# scaler.fit(X)
# X=scaler.transform(X)
# Make dataseta
X_train, X_test, z_train, z_test = train_test_split(X, y, test_size=0.2) #, random_state=42)

GD = GradientDescent(X_train, X_test, z_train, z_test, eta=0.0001, epochs=10 ,batch_size=10)
beta_sgd = GD.stochastic_gradient_descent(method='OLS', learning_schedule='plain')
beta_gd = GD.gradient_descent(method='OLS', learning_schedule='plain')
beta_sgd_jax = GD.stochastic_gradient_descent_jax(method='OLS', learning_schedule='plain')
beta_gd_jax = GD.gradient_descent_jax(method='OLS', learning_schedule='plain')
sgd_solution = X@ beta_sgd
sgd_solution_jax = X@ beta_sgd_jax
gd_solution = X@beta_gd
gd_solution_jax = X@beta_gd_jax
fig1 = plt.figure()
plt.plot(x, y, 'r.', label='Data')
plt.plot(x, 1+x+ x**2, '-b', label='True function')
plt.plot(x, sgd_solution, '--g', label='SGD')
plt.plot(x, gd_solution, '--m', label='GD')
plt.grid()
plt.legend()
plt.title('Plain OLS with JAX gradient')
plt.savefig('jax_solution_sgd_vs_gd.png')

# fig2 = plt.figure()
# #plt.plot(x, y, 'r.', label='Data')
# plt.plot(x, 1+x+ x**2, '-b', label='True function')
# plt.plot(x, sgd_solution, '-g', label='SGD')
# plt.plot(x, sgd_solution_jax, '--g', label='SGD JAX')
# plt.plot(x, gd_solution, '-m', label='GD')
# plt.plot(x, gd_solution_jax, '--m', label='GD JAX')
# plt.grid(); plt.legend()
# plt.title('Analytical vs JAX gradient')
# plt.savefig('jax_vs_analytical.png')


# find best eta
eta_values = np.logspace(-10, 4, 30)
lmbd=0.001
batch_size = 10
epochs={}
eta = {}
mse = {}
r2 = {} 
methods = {}
beta = {}
n_batches = {}

scores = { 'Method': methods, 'Eta': eta, 'MSE': mse,'R2': r2,'Beta': beta, 'Number of epochs': epochs, 'Number of batches': n_batches }
#make dict with all values, not so pretty when adding 1+number to get a unique row number for all values
for i, eta in enumerate(eta_values): 
    t = time.time()
    GD = GradientDescent(X_train, X_test, z_train, z_test, eta=eta, epochs=10 ,batch_size=batch_size, lmbd=None) 
    scores['Eta'][i] = eta  
    scores['Number of epochs'][i]=epochs
    scores['Beta'][i] = GD.stochastic_gradient_descent(method='Ridge', learning_schedule='plain')
    scores['MSE'][i] = GD.MSE()
    scores['R2'][i] = GD.R2()
    scores['Method'][i] = 'SGD plain'
    # # Momentum
    i = i+ len(eta_values) 
    scores['Eta'][i] = eta  
    scores['Number of epochs'][i]=epochs
    scores['Beta'][i] = GD.stochastic_gradient_descent(method='OLS', learning_schedule='momentum')
    scores['MSE'][i] = GD.MSE()
    scores['R2'][i] = GD.R2()
    scores['Method'][i] = 'SGD momentum'
    # Adagrad 
    i = i+ 2*len(eta_values) 
    scores['Eta'][i] = eta  
    scores['Number of epochs'][i]=epochs
    scores['Beta'][i] = GD.stochastic_gradient_descent(method='OLS', learning_schedule='adagrad')
    scores['MSE'][i] = GD.MSE()
    scores['R2'][i] = GD.R2()
    scores['Method'][i] = 'SGD adagrad'
    # ADam
    i = i+ 3*len(eta_values) 
    scores['Eta'][i] = eta  
    scores['Number of epochs'][i]=epochs
    scores['Beta'][i] = GD.stochastic_gradient_descent(method='OLS', learning_schedule='adam')
    scores['MSE'][i] = GD.MSE()
    scores['R2'][i] = GD.R2()
    scores['Method'][i] = 'SGD adam'
    #RMS-prop
    i = i+ 4*len(eta_values) 
    scores['Eta'][i] = eta  
    scores['Number of epochs'][i]=epochs
    scores['Beta'][i] = GD.stochastic_gradient_descent(method='OLS', learning_schedule='rms_prop')
    scores['MSE'][i] = GD.MSE()
    scores['R2'][i] = GD.R2()
    scores['Method'][i] = 'SGD rms-prop'
    elapsed = time.time() - t
    print(f'Time for round {i-299} of {len(eta_values)} with eta={eta}: {elapsed}')

fig1 = px.line(scores, x="Eta", y="R2", title='R2',color='Method',log_x=True)
fig1.update_traces(mode='lines+markers')

fig1.update_layout(
    title = 'Prediction error using various learning schedules',
    xaxis_title="\u03B7",
    xaxis = dict(
        showexponent = 'all',
        exponentformat = 'e'
    )
)


fig1.show()
# fig1.write_image('sgd_eta_ridge.png', width=1920, height=1080)



# compare with scikit learn
eta = 1e-5
batch_size = 20

GD = GradientDescent(X_train, X_test, z_train, z_test, eta=eta, epochs=50 ,batch_size=batch_size, lmbd='None')
beta = GD.stochastic_gradient_descent(method='OLS', learning_schedule='plain')
r2=GD.R2()
# scaler = StandardScaler()
# scaler.fit(X)
# X=scaler.transform(X)
clf = linear_model.SGDRegressor(max_iter=50,  eta0=eta)
clf.fit(X_train, z_train)
clf.score(X_test, z_test)
