import numpy as np
from random import random, seed
from IPython import embed
#from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from sklearn import linear_model
import time


def ols(X,y):
    beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y) # Find beta with matrix inversion (pseudoinverse)
    return beta 

def ridge(X, y, lambd):    
    size_train = X.shape[1] # Size for identity matrix   
    beta = np.linalg.pinv( X.T @ X + lambd * np.identity(size_train)) @ X.T @ y   
    return beta 


def MSE(z_pred, z_test):
    mse = np.sum((np.subtract(z_test,z_pred))**2 )* (1/len(z_test))
    return mse

def R2( z_pred, z_test):
    z_mean = np.mean(z_test)
    R2 = 1- (np.sum((z_test-z_pred)**2 ) / (np.sum((z_test- z_mean)**2 )))
    return R2

def franke_function(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2) 
    noise = 0.05 * np.random.randn(x.shape[0], x.shape[1])  # add noise
    z = term1 + term2 + term3 + term4 + noise
    return z

def create_design_matrix(x,y,m):
    x = np.ravel(x)                                     # Return flattened x array
    y = np.ravel(y)                                     # Return flattened y array
    N = len(x)
    l = int((m+1)*(m+2)/2)		                        # Number of elements in beta
    X = np.ones((N,l))                                  # Initiate design matrix with only ones
    for i in range(1,m+1):                              # Enter loop to fill up each column of design matrix, starting from the 1st order position (because 0th order = 1)
        q = int((i)*(i+1)/2)                            # q = order of each column
        for k in range(i+1):                            # 
            X[:,q+k] = x**(i-k) * y**k                  # Fill upp all columns of design matrix with variable of the right order

    X_no_intcept=X[:,1:] 
    return X_no_intcept

def create_design_matrix_with_intercept(x,y,m):
    x = np.ravel(x)                                     # Return flattened x array
    y = np.ravel(y)                                     # Return flattened y array
    N = len(x)
    l = int((m+1)*(m+2)/2)		                        # Number of elements in beta
    X = np.ones((N,l))                                  # Initiate design matrix with only ones
    for i in range(1,m+1):                              # Enter loop to fill up each column of design matrix, starting from the 1st order position (because 0th order = 1)
        q = int((i)*(i+1)/2)                            # q = order of each column
        for k in range(i+1):                            # 
            X[:,q+k] = x**(i-k) * y**k                  # Fill upp all columns of design matrix with variable of the right order
    return X

# Cross validation function
def cross_validation(X, z, folds, model, lambd):
    t = time.time()                 # benchmark time used
    np.random.seed(2018)
    n = len(z)
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // folds              # returns integer
    mse = np.zeros(folds)
    r2 = np.zeros(folds)
    z_pred = np.zeros((folds, fold_size))
  
    for i in range(folds):
        # Make indices for dividing data into folds
        start = i * fold_size
        stop = (i+1) * fold_size
        val_indices = indices[start:stop]                                       # Make test indices
        train_indices = np.concatenate((indices[:start], indices[stop:]))       # Use the rest for training
       
       # Use indices to pick out data for training and testing
        X_train = X[train_indices]      
        z_train = z[train_indices]
        X_test = X[val_indices]
        z_test = z[val_indices]

        # Scaling
        X_train_mean = X_train.mean(axis=0)[np.newaxis,:]
        z_train_mean = z_train.mean()
        X_train = X_train-X_train_mean
        X_test = X_test-X_train_mean
        z_train = z_train-z_train_mean

        # Predict z and find error
        if model=='ols':
            beta = ols(X_train, z_train)
            intercept = np.mean(z_train_mean - X_train_mean @ beta)
            z_pred[i,:] = X_test @ beta + z_train_mean
        elif model=='ridge':
            beta = ridge(X_train, z_train, lambd)
            intercept = np.mean(z_train_mean - X_train_mean @ beta)
            z_pred[i,:] = X_test @ beta + z_train_mean
        mse[i] = MSE(z_pred[i,:], z_test)
        r2[i] = R2(z_pred[i,:], z_test)
    mse_mean = np.mean(mse)
    r2_mean = np.mean(r2)
    elapsed = time.time() - t
    print(f'Time used for cross-validation with {folds} folds: {elapsed}')
    return mse_mean, z_pred, r2_mean

# Cross validation function using scikit
def cross_validation_scikit(X, z, folds, model, lambd):
    np.random.seed(2018)
    n = len(z)
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // folds              # returns integer
    mse = np.zeros(folds)
    r2 = np.zeros(folds)
    z_pred = np.zeros((folds, fold_size))
  
    for i in range(folds):
        # Make indices for dividing data into folds
        start = i * fold_size
        stop = (i+1) * fold_size
        val_indices = indices[start:stop]                                       # Make test indices
        train_indices = np.concatenate((indices[:start], indices[stop:]))       # Use the rest for training
       
       # Use indices to pick out data for training and testing
        X_train = X[train_indices]      
        z_train = z[train_indices]
        X_test = X[val_indices]
        z_test = z[val_indices]
      
        # Scaling
        X_train_mean = X_train.mean(axis=0)[np.newaxis,:]
        z_train_mean = z_train.mean()
        X_train = X_train-X_train_mean
        X_test = X_test-X_train_mean
        z_train = z_train-z_train_mean
        z_test_scaled = z_test-z_train_mean

        # Predict z and find error
        if model=='ols':
            olsfit = linear_model.LinearRegression().fit(X_train, z_train)
            beta = olsfit.coef_
            z_pred[i,:] = X_test @ beta 
        elif model=='ridge':
            ridgefit = linear_model.Ridge(alpha=lambd, fit_intercept=False).fit(X_train, z_train)
            beta = ridgefit.coef_
            z_pred[i,:] = X_test @ beta 
        elif model=='lasso':
            lassofit = linear_model.Lasso(alpha=lambd,fit_intercept=False, tol=0.0001).fit(X_train, z_train)
            beta = lassofit.coef_
            z_pred[i,:]= X_test @ beta
        r2[i] = R2(z_pred[i,:], z_test_scaled)
        mse[i] = MSE(z_pred[i,:], z_test_scaled)
    mse_mean = np.mean(mse)
    r2_mean = np.mean(r2)
    return mse_mean, z_pred , r2_mean

def bootstrap(X, z, n_bootstraps, model, lambd):
    t = time.time()                 # benchmark time
    # Make test and train data
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, random_state=42)

    # Scaling
    X_train_mean = X_train.mean(axis=0)[np.newaxis,:]
    z_train_mean = z_train.mean()
    X_train = X_train-X_train_mean
    X_test = X_test-X_train_mean
    z_train = z_train-z_train_mean

    # Solution vectors
    z_pred = np.zeros((n_bootstraps,z_test.shape[0]))
    mse = np.zeros(n_bootstraps)

    for i in range(int(n_bootstraps)):
        # Draw samples with replacement from training data
        X_boot, z_boot = resample(X_train, z_train)   
    
        # Predict z and find error
        if model=='ols':
            beta = ols(X_boot, z_boot)
            intercept = np.mean(z_train_mean - X_train_mean @ beta)
            z_pred[i,:] = X_test @ beta + z_train_mean
        elif model=='ridge':
            beta = ridge(X_boot, z_boot, lambd)
            intercept = np.mean(z_train_mean - X_train_mean @ beta)
            z_pred[i,:] = X_test @ beta+ z_train_mean      
        mse[i] = MSE(z_pred[i,:], z_test)    
    mse_mean = np.mean(mse)
    elapsed = time.time() - t
    print(f'Time used for bootstrap with {n_bootstraps} samples: {elapsed}')
    return mse_mean, z_pred

def bootstrap_scikit(X, z, n_bootstraps, model, lambd):
    # Make test and train data
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, random_state=42)

    # Scaling
    X_train_mean = X_train.mean(axis=0)[np.newaxis,:]
    z_train_mean = z_train.mean()
    X_train = X_train-X_train_mean
    X_test = X_test-X_train_mean
    z_train = z_train-z_train_mean
    z_test_scaled = z_test-z_train_mean

    # Solution vectors
    z_pred = np.zeros((n_bootstraps,z_test.shape[0]))
    mse = np.zeros(n_bootstraps)

    for i in range(int(n_bootstraps)):
        # Draw samples with replacement from training data
        X_boot, z_boot = resample(X_train, z_train)   

        # Predict z and find error
        if model=='ols':
            olsfit = linear_model.LinearRegression().fit(X_boot, z_boot)
            beta = olsfit.coef_
            z_pred[i,:] = X_test @ beta 
        elif model=='ridge':
            ridgefit = linear_model.Ridge(alpha=lambd, fit_intercept=False).fit(X_boot, z_boot)
            beta = ridgefit.coef_
            z_pred[i,:] = X_test @ beta 
        elif model=='lasso':
            lassofit = linear_model.Lasso(alpha=lambd,fit_intercept=False, tol=0.0001, max_iter = 2000).fit(X_boot, z_boot) # no_intercept = false because data already scaled
            beta = lassofit.coef_
            z_pred[i,:]= X_test @ beta      
        mse[i] = mean_squared_error(z_pred[i,:], z_test_scaled)   
    mse_mean = np.mean(mse)
    return mse_mean, z_pred