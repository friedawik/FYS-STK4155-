from utils import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
from IPython import embed
import pandas as pd
from matplotlib import cm
import matplotlib as mpl
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import resample
from imageio import imread 
from matplotlib.ticker import MaxNLocator


# Make data
np.random.seed(2018)
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)
z_ = franke_function(x,y)                    # unraveled
z = np.ravel(z_)
m = 2                                  # Order of polynomials                   

# Design matrix
X = create_design_matrix(x,y,m)
X_intercept = create_design_matrix_with_intercept(x,y,m)

# Cross validation parameters
folds = 5

# Bootstrap parameters
test_part = 0.2
n_bootstraps = 100

# Ridge and lasso regularisation
lambd=1e-4

# Print to check that behaves as scikit learn
print('Cross validation OLS')
print(cross_validation(X, z, folds, model='ols', lambd=None)[0])
print('Cross validation OLS Scikit')
print(cross_validation_scikit(X, z, folds, model='ols', lambd=None)[0])
print('Cross validation ridge')
print(cross_validation(X, z, folds, model='ridge', lambd= lambd )[0])
print('Cross validation ridge Scikit')
print(cross_validation_scikit(X, z, folds, model='ridge', lambd= lambd )[0])
print('Cross validation lasso')
print(cross_validation_scikit(X, z, folds, model='lasso', lambd=lambd)[0])

my_bootstrap = bootstrap(X, z, n_bootstraps, model='ols', lambd=None)[0]
scikit_bootstrap = bootstrap_scikit(X_intercept, z, n_bootstraps, model='ols', lambd=None)[0]
print('Bootstrap OLS')
print(my_bootstrap)
print('Bootstrap OLS Scikit')
print(scikit_bootstrap)


#  #Plot to compare bootstrap and cv with diff lambda
lambd = np.logspace(-8, 8, 50)          # penalization values for lasso and ridge
# Cross validation parameters
folds = 5
mse_cv_ridge = np.zeros((len(lambd)))
mse_cv_ridge_scikit = np.zeros((len(lambd)))

# Bootstrap parameters
test_part = 0.2
n_bootstraps = 30
mse_bootstrap_ridge = np.zeros((len(lambd)))
mse_bootstrap_ridge_scikit = np.zeros((len(lambd)))

# Degree
m=5

# Design matrix
X = create_design_matrix(x,y,m)
  
for i, lam in enumerate(lambd):
    mse_bootstrap_ridge[i] = bootstrap(X, z, n_bootstraps,'ridge', lam)[0]
    mse_bootstrap_ridge_scikit[i] = bootstrap_scikit(X, z, n_bootstraps, 'ridge', lam)[0]
    mse_cv_ridge[i]=cross_validation(X, z, folds, 'ridge', lam)[0]
    mse_cv_ridge_scikit[i]=cross_validation_scikit(X, z, folds, 'ridge', lam)[0]

plt.plot(np.log10(lambd),mse_bootstrap_ridge ,'g-',label= 'My bootstrap')
plt.plot(np.log10(lambd),mse_bootstrap_ridge_scikit, 'g--', label='Scikit bootstrap')
plt.plot(np.log10(lambd),mse_cv_ridge, 'b-', label= 'My CV')
plt.plot(np.log10(lambd),mse_cv_ridge_scikit, 'b--', label='Scikit CV')

plt.xlabel('log10($\lambda$)')
plt.ylabel('Mean squared error (MSE)')
plt.grid()
plt.legend()
plt.title('Bootstrap and CV: Scikit vs. own code')
plt.savefig('scikit_bootstrap_vs_cross_validation_diff_lambda.png')

my_bootstrap = bootstrap(X, z, n_bootstraps, model='ols', lambd=None)[0]
scikit_bootstrap = bootstrap_scikit(X_intercept, z, n_bootstraps, model='ols', lambd=None)[0]
print('Bootstrap OLS')
print(my_bootstrap)
print('Bootstrap OLS Scikit')
print(scikit_bootstrap)

my_bootstrap = bootstrap(X, z, n_bootstraps, model='ridge', lambd=1e-4)[0]
scikit_bootstrap = bootstrap_scikit(X, z, n_bootstraps, model='ridge', lambd=1e-4)[0]
print('Bootstrap Ridge')
print(my_bootstrap)
print('Bootstrap Ridge Scikit')
print(scikit_bootstrap)

#####################
#Make Bias-variance trade-off plot with bootstrap

# Vectors to store values
polydegree = np.arange(1,6,1)
error_own = np.zeros(len(polydegree))
error_scikit = np.zeros(len(polydegree))
bias_own = np.zeros(len(polydegree))
bias_scikit = np.zeros(len(polydegree))
variance_own = np.zeros(len(polydegree))
variance_scikit = np.zeros(len(polydegree))

for d, deg in enumerate(polydegree):
    X = create_design_matrix(x,y,deg)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=test_part, random_state=42)
    # Scaling
    X_train_mean = X_train.mean(axis=0)[np.newaxis,:]
    z_train_mean = z_train.mean()
    X_train = X_train-X_train_mean
    X_test = X_test-X_train_mean
    z_train = z_train-z_train_mean
    z_test_scaled = z_test-z_train_mean

    z_pred_own = np.zeros((z_test.shape[0], int(n_bootstraps)))
    z_pred_scikit = np.zeros((z_test.shape[0], int(n_bootstraps)))
    z_pred_lasso = np.zeros((z_test.shape[0], int(n_bootstraps)))

    for i in range(int(n_bootstraps)):       # bootstrapping        
        X_boot, z_boot = resample(X_train, z_train) 
        beta_ols = ols(X_boot, z_boot)
        z_pred_own[:,i] = X_test @ beta_ols + z_train_mean
        olsfit = linear_model.LinearRegression(fit_intercept=False).fit(X_boot, z_boot)
        beta_ols = olsfit.coef_
        z_pred_scikit[:,i] =X_test @ beta_ols
              

    # compute errors. Adding a dimenson to test vector needed to compute correctly the mean over axis 1 [:, np.newaxis]
    error_own[d] = np.mean( np.mean((z_test[:, np.newaxis] - z_pred_own)**2, axis=1, keepdims=True ))
    error_scikit[d] = np.mean( np.mean((z_test_scaled[:, np.newaxis] - z_pred_scikit)**2, axis=1, keepdims=True) )
    bias_own[d] = np.mean( (z_test[:, np.newaxis] - np.mean(z_pred_own, axis=1, keepdims=True))**2 )
    bias_scikit[d] = np.mean( (z_test_scaled[:, np.newaxis] - np.mean(z_pred_scikit, axis=1, keepdims=True))**2)    
    variance_own[d] = np.mean( np.var(z_pred_own, axis=1, keepdims=True) )
    variance_scikit[d] = np.mean( np.var(z_pred_scikit, axis=1, keepdims=True))

   
    print('OLS:')
    print('Error:', error_own[d])
    print('Bias^2:', bias_own[d])
    print('Var:', variance_own[d])
    print('{} >= {} + {} = {}'.format(error_own[d], bias_own[d], variance_own[d], bias_own[d]+variance_own[d]))
    print('Ridge:')
    print('Error:', error_scikit[d])
    print('Bias^2:', bias_scikit[d])
    print('Var:', variance_scikit[d])
    print('{} >= {} + {} = {}'.format(error_scikit[d], bias_scikit[d], variance_scikit[d], bias_scikit[d]+variance_scikit[d]))

fig1=plt.figure()
plt.plot(polydegree, error_own, label='Error')
plt.plot(polydegree, bias_own, label='Squared bias')
plt.plot(polydegree, variance_own, label='Variance')
plt.plot(polydegree, error_scikit,'--', label='Error scikit')
plt.plot(polydegree, bias_scikit,'--', label='Squared bias scikit')
plt.plot(polydegree, variance_scikit,'--', label='Variance scikit')
plt.xticks(polydegree)
plt.grid()
plt.xlabel('Polynomial degree')
plt.ylabel('Error')
plt.title('Bias-variance trade-off: Scikit vs. own code')
plt.legend()
plt.savefig('scikit_bias_variance_tradeoff.png')

