import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
from IPython import embed
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import resample
import imageio.v2 as imageio
import xarray as xr
from utils import *
from matplotlib.ticker import MaxNLocator

# # Make data
terrain = imageio.imread('SRTM_data_Norway_1.tif')
z_squared = terrain[1800:3600, 0:1800]             # take out a square of the data
z_small = z_squared[::10, ::10]                    # use every tenth pixel
#z_small = terrain[::40, ::40]

#Scaling - to make dataset smaller and make z,x,y on same scale
xydist = 55e3 # meters. approx. distance between 8-9 W degree at 60 N
x = np. linspace(0, 1,z_small.shape[1])
y = np. linspace(0, 1,z_small.shape[0])
x, y = np.meshgrid(x,y)
z_small=z_small* (1/xydist)
z = np.ravel(z_small)
print(z_small.min(), z_small.max())

# x = np. linspace(0, 1,z_small.shape[1])
# y = np. linspace(0, 1,z_small.shape[0])
# x, y = np.meshgrid(x,y)
# z_small=z_small/z_small.max()
# z = np.ravel(z_small)
# print(z.min(), z.max())
# terrain = imageio.imread('SRTM_data_Norway_1.tif')

# N = 1000
# terrain = terrain[:N,:N]
# z_small = terrain[::10, ::10] 
# embed()
# # Creates mesh of image pixels
# x_ = np.linspace(0,1, np.shape(terrain)[0])
# y_ = np.linspace(0,1, np.shape(terrain)[1])
# x, y = np.meshgrid(x_,y_)

# z = np.ravel(terrain)
#X = create_X(x_mesh, y_mesh,m)

# z = np.array(terrain)
# z_small = z[0:100, 0:100]             # take out parts of data
# z_small = z[::40, ::40]
# #z_small = z
# x = np.arange(z_small.shape[1])
# y = np.arange(z_small.shape[0])
# x, y = np.meshgrid(x,y)
# z = np.ravel(z_small)


#print(z_small.min(), z_small.max())


#  Plot to compare bootstrap and cv with diff lambda
# Parameters
m=20 # degree
X = create_design_matrix(x,y,m)
lambd = np.logspace(-8, 8, 50)          # penalization values for lasso and ridge
folds = 5           # for cv
n_bootstraps = 30   # for bootstrap
# solution vectors
mse_cv_ridge = np.zeros((len(lambd)))
mse_cv_lasso = np.zeros((len(lambd)))
mse_bootstrap_ridge = np.zeros((len(lambd)))
mse_bootstrap_lasso = np.zeros((len(lambd)))
for i, lam in enumerate(lambd):
    mse_bootstrap_ridge[i] = bootstrap(X, z, n_bootstraps,'ridge', lam)[0]
    mse_bootstrap_lasso[i] = bootstrap_scikit(X, z, n_bootstraps, 'lasso', lam)[0]
    
    mse_cv_ridge[i]=cross_validation(X, z, folds, 'ridge', lam)[0]
    mse_cv_lasso[i]=cross_validation_scikit(X, z, folds, 'lasso', lam)[0]

plt.plot(np.log10(lambd),mse_bootstrap_ridge ,'g-',label= 'Bootstrap ridge')
plt.plot(np.log10(lambd),mse_cv_ridge, 'g--', label='CV ridge')
plt.plot(np.log10(lambd),mse_bootstrap_lasso, 'b-', label= 'Bootstrap lasso')
plt.plot(np.log10(lambd),mse_cv_lasso, 'b--', label='CV lasso')

plt.xlabel('log10($\lambda$)')
plt.ylabel('Mean squared error (MSE)')
plt.grid()
plt.legend()
plt.title('Screening for best "$\lambda$"')
plt.savefig('terrain_bootstrap_vs_cross_validation_diff_lambda.png')

mse_ols = cross_validation(X, z, folds, 'ols', lambd=None)[0]
print(mse_ols)

# Plot to choose different folds in cv 
folds = np.arange(5, 11, 1)          # folds from 5 to 10
# Cross validation parameters
mse_cv_ols = np.zeros((len(folds)))
mse_cv_ridge = np.zeros((len(folds)))
mse_cv_lasso = np.zeros((len(folds)))

# Degree
m=12

# regularisation
lambd = 1e-5

# Design matrix
X = create_design_matrix(x,y,m)
 
for i, fol in enumerate(folds):    
    mse_cv_ols[i]=cross_validation(X, z, fol, 'ols', lambd=None)[0]
    mse_cv_ridge[i]=cross_validation(X, z, fol, 'ridge', lambd)[0]
    mse_cv_lasso[i]=cross_validation_scikit(X, z, fol, 'lasso', lambd)[0]
    
fig, ax = plt.subplots()
ax.plot(folds,mse_cv_ols, 'r-', label='OLS')
ax.plot(folds,mse_cv_ridge, 'g--', label='Ridge')
ax.plot(folds,mse_cv_lasso, 'b--', label='Lasso')

plt.xlabel('Folds')
plt.ylabel('Mean squared error (MSE)')
plt.grid()
plt.legend()
plt.title('Testing fold size')
plt.savefig('terrain_cross_validation_diff_folds.png')

 #Plot to compare bootstrap and cv with diff degree
degree = np.arange(1,26,1)        # degree
# Cross validation parameters
folds = 5
mse_cv_ols = np.zeros(len(degree))
mse_cv_ridge = np.zeros(len(degree))
mse_cv_lasso = np.zeros(len(degree))

# Bootstrap parameters
n_bootstraps = 30
mse_bootstrap_ols = np.zeros(len(degree))
mse_bootstrap_ridge = np.zeros(len(degree))
mse_bootstrap_lasso = np.zeros(len(degree))

# Regularisation
lambd=1e-4

for i, deg in enumerate(degree):
    X = create_design_matrix(x,y,deg)
    mse_bootstrap_ols[i] = bootstrap(X, z, n_bootstraps,'ols', lambd=None)[0]
    mse_bootstrap_ridge[i] = bootstrap(X, z, n_bootstraps,'ridge', lambd)[0]
    mse_bootstrap_lasso[i] = bootstrap_scikit(X, z, n_bootstraps, 'lasso', lambd)[0]
    mse_cv_ols[i]=cross_validation(X, z, folds, 'ols', lambd=None)[0] 
    mse_cv_ridge[i]=cross_validation(X, z, folds, 'ridge', lambd)[0]
    mse_cv_lasso[i]=cross_validation_scikit(X, z, folds, 'lasso', lambd)[0]

fig, ax = plt.subplots()
ax.plot(degree,mse_bootstrap_ols ,'r-',label= 'Bootstrap ols')
ax.plot(degree,mse_cv_ols, 'r--', label='CV ols')
ax.plot(degree,mse_bootstrap_ridge ,'g-',label= 'Bootstrap ridge')
ax.plot(degree,mse_cv_ridge, 'g--', label='CV ridge')
ax.plot(degree,mse_bootstrap_lasso, 'b-', label= 'Bootstrap lasso')
ax.plot(degree,mse_cv_lasso, 'b--', label='CV lasso')

ax.set_xlabel('Polynomial degree')
ax.set_ylabel('Mean squared error (MSE)')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.grid()
plt.legend()
plt.savefig('terrain_bootstrap_vs_cross_validation_degree.png')

##################
#Make Bias variance trade off plot with bootstrap

# parameters
polydegree = np.arange(1,21,1)
lambd = 1e-5
n_bootstraps = 30
# Vectors to store values
error_ols = np.zeros(len(polydegree))
error_ridge = np.zeros(len(polydegree))
error_lasso = np.zeros(len(polydegree))
bias_ols = np.zeros(len(polydegree))
bias_ridge = np.zeros(len(polydegree))
bias_lasso = np.zeros(len(polydegree))
variance_ols = np.zeros(len(polydegree))
variance_ridge = np.zeros(len(polydegree))
variance_lasso = np.zeros(len(polydegree))


for d, deg in enumerate(polydegree):

    X = create_design_matrix(x,y,deg)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, random_state=42)

    # Scaling
    X_train_mean = X_train.mean(axis=0)[np.newaxis,:]
    z_train_mean = z_train.mean()
    X_train = X_train-X_train_mean
    X_test = X_test-X_train_mean
    z_train = z_train-z_train_mean
    z_test_scaled = z_test-z_train_mean

    z_pred_ols = np.zeros((z_test.shape[0], int(n_bootstraps)))
    z_pred_ridge = np.zeros((z_test.shape[0], int(n_bootstraps)))
    z_pred_lasso = np.zeros((z_test.shape[0], int(n_bootstraps)))


    for i in range(int(n_bootstraps)):       # bootstrapping        
        X_boot, z_boot = resample(X_train, z_train) 
        beta_ols = ols(X_boot, z_boot)
        z_pred_ols[:,i] = X_test @ beta_ols + z_train_mean
        beta_ridge = ridge(X_boot, z_boot, lambd)
        z_pred_ridge[:,i] = X_test @ beta_ridge + z_train_mean

        lassofit = linear_model.Lasso(alpha=lambd, fit_intercept=False).fit(X_boot, z_boot)
        beta_lasso = lassofit.coef_
        z_pred_lasso[:,i] =X_test @ beta_lasso
              

    # compute errors. Adding a dimenson to test vector needed to compute correctly the mean over axis 1 [:, np.newaxis]
    error_ols[d] = np.mean( np.mean((z_test[:, np.newaxis] - z_pred_ols)**2, axis=1, keepdims=True ))
    error_ridge[d] = np.mean( np.mean((z_test[:, np.newaxis] - z_pred_ridge)**2, axis=1, keepdims=True) )
    error_lasso[d] = np.mean( np.mean((z_test_scaled[:, np.newaxis] - z_pred_lasso)**2, axis=1, keepdims=True) )
    bias_ols[d] = np.mean( (z_test[:, np.newaxis] - np.mean(z_pred_ols, axis=1, keepdims=True))**2 )
    bias_ridge[d] = np.mean( (z_test[:, np.newaxis] - np.mean(z_pred_ridge, axis=1, keepdims=True))**2)
    bias_lasso[d] = np.mean( (z_test_scaled[:, np.newaxis] - np.mean(z_pred_lasso, axis=1, keepdims=True))**2)       
    variance_ols[d] = np.mean( np.var(z_pred_ols, axis=1, keepdims=True) )
    variance_ridge[d] = np.mean( np.var(z_pred_ridge, axis=1, keepdims=True))
    variance_lasso[d] = np.mean( np.var(z_pred_lasso, axis=1, keepdims=True) )
   
    print('OLS:')
    print('Error:', error_ols[d])
    print('Bias^2:', bias_ols[d])
    print('Var:', variance_ols[d])
    print('{} >= {} + {} = {}'.format(error_ols[d], bias_ols[d], variance_ols[d], bias_ols[d]+variance_ols[d]))
    print('Ridge:')
    print('Error:', error_ridge[d])
    print('Bias^2:', bias_ridge[d])
    print('Var:', variance_ridge[d])
    print('{} >= {} + {} = {}'.format(error_ridge[d], bias_ridge[d], variance_ridge[d], bias_ridge[d]+variance_ridge[d]))
    print('Lasso:')
    print('Error:', error_lasso[d])
    print('Bias^2:', bias_lasso[d])
    print('Var:', variance_lasso[d])
    print('{} >= {} + {} = {}'.format(error_lasso[d], bias_lasso[d], variance_lasso[d], bias_lasso[d]+variance_lasso[d]))


fig1=plt.figure()
plt.plot(polydegree, error_ols, label='Error')
plt.plot(polydegree, bias_ols,'--', label='Squared bias')
plt.plot(polydegree, variance_ols, label='Variance')
plt.xticks(polydegree)
plt.grid()
plt.xlabel('Polynomial degree')
plt.ylabel('Error')
plt.title('OLS')
#plt.ylim(top=60000)
plt.legend()
plt.savefig('terrain_bias_variance_tradeoff_ols.png')

fig2=plt.figure()
plt.plot(polydegree, error_ridge, label='Error')
plt.plot(polydegree, bias_ridge,'--', label='Squared bias')
plt.plot(polydegree, variance_ridge, label='Variance')
plt.xticks(polydegree)
plt.grid()
#plt.ylim(top=60000)
plt.xlabel('Polynomial degree')
plt.ylabel('Error')
plt.title('Ridge')
plt.legend()
plt.savefig('terrain_bias_variance_tradeoff_ridge.png')

fig3=plt.figure()
plt.plot(polydegree, error_lasso, label='Error')
plt.plot(polydegree, bias_lasso, '--',label='Squared bias')
plt.plot(polydegree, variance_lasso, label='Variance')
plt.xticks(polydegree)
#plt.ylim(top=60000)
plt.grid()
plt.xlabel('Polynomial degree')
plt.ylabel('Error')
plt.title('Lasso')
plt.legend()
plt.savefig('terrain_bias_variance_tradeoff_lasso.png')


# Beta plots OLS
degree = np.arange(1,21,1)
fig = plt.figure()
for i, deg in enumerate(degree):
    X = create_design_matrix(x,y,deg)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, random_state=42)
   
    beta_ols  = ols(X_train, z_train)
    x_plot= deg*np.ones(beta_ols.shape)
    plt.scatter(x_plot,beta_ols, c='#d62728')

plt.xlabel('Polynomial degree')
plt.ylabel(r'$\beta$')
plt.grid()
plt.legend()
plt.yscale('symlog')
plt.title('Optimal coefficients for OLS regression')
plt.ylim(-1e4,1e4)
plt.savefig('terrain_beta_ols.png')

# Beta ridge plots
lambd = 1e-5
fig = plt.figure()
for i, deg in enumerate(degree):
        
    X = create_design_matrix(x,y,deg)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, random_state=42)
   
    beta_ridge  = ridge(X_train, z_train, lambd)
    x_plot= deg*np.ones(beta_ridge.shape)
    plt.scatter(x_plot,beta_ridge, c='#d62728')

plt.xlabel('Polynomial degree')
plt.ylabel(r'$\beta$')
plt.grid()
plt.yscale('symlog')
plt.title('Optimal coefficients for Ridge regression')
plt.ylim(-1e4,1e4)
plt.savefig('terrain_beta_ridge.png')

# Beta lasso plots
fig = plt.figure()
for i, deg in enumerate(degree):
        
    X = create_design_matrix(x,y,deg)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, random_state=42)
    lassofit = linear_model.Lasso(alpha=lambd,fit_intercept=False, tol=0.0001).fit(X_train, z_train)
    beta_lasso = lassofit.coef_
    x_plot= deg*np.ones(beta_lasso.shape)
    plt.scatter(x_plot,beta_lasso, c='#d62728')
plt.xlabel('Polynomial degree')
plt.ylabel(r'$\beta$')
plt.yscale('symlog')
plt.grid()
plt.legend()
plt.title('Optimal coefficients for Lasso regression')
plt.ylim(-1e4,1e4)
plt.savefig('terrain_beta_lasso.png')

 #Plot r2 from cv with diff degree
degree = np.arange(1,21,1)        # degree
lambd = 1e-6
# Cross validation parameters
folds = 5
r2_ols = np.zeros(len(degree))
r2_ridge = np.zeros(len(degree))
r2_lasso = np.zeros(len(degree))

for i, deg in enumerate(degree):
    X = create_design_matrix(x,y,deg)
    r2_ols[i]=cross_validation(X, z, folds,  'ols', lambd=None)[2] 
    r2_ridge[i]=cross_validation(X, z, folds, 'ridge', lambd)[2] 
    r2_lasso[i]=cross_validation_scikit(X, z, folds, 'lasso', lambd)[2] 

fig, ax = plt.subplots()
ax.plot(degree,r2_ols ,label= 'OLS')
ax.plot(degree,r2_ridge, '--',label='Ridge')
ax.plot(degree,r2_lasso ,label= 'Lasso')
ax.set_xlabel('Polynomial degree')
ax.set_ylabel('R2 score')
ax.set_title('R2 score')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.grid()
plt.legend()
plt.savefig('terrain_r2_degree.png')

 #Plot mse from cv with diff degree
lambd = 1e-6
degree = np.arange(1,21,1)        # degree
folds = 5
mse_ols = np.zeros(len(degree))
mse_ridge = np.zeros(len(degree))
mse_lasso = np.zeros(len(degree))

for i, deg in enumerate(degree):
    X = create_design_matrix(x,y,deg)
    mse_ols[i]=cross_validation(X, z, folds,  'ols', lambd=None)[0] 
    mse_ridge[i]=cross_validation(X, z, folds, 'ridge', lambd)[0] 
    mse_lasso[i]=cross_validation_scikit(X, z, folds, 'lasso', lambd)[0] 

fig, ax = plt.subplots()
ax.plot(degree,mse_ols ,label= 'OLS')
ax.plot(degree,mse_ridge, '--',label='Ridge')
ax.plot(degree, mse_lasso ,label= 'Lasso')
ax.set_xlabel('Polynomial degree')
ax.set_ylabel('Mean squared error (MSE)')
ax.set_title('MSE')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.grid()
plt.legend()
plt.savefig('terrain_mse_degree.png')

# Figure mse train test   
m =41                              # Order of polynomials
degree = np.arange(m)
mse_test = np.zeros(m)
mse_train = np.zeros(m)

for i in degree:
    X = create_design_matrix(x,y, i)
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, random_state=42)
    # Scaling
    X_train_mean = X_train.mean(axis=0)[np.newaxis,:] # add dimension to be able to subtract later
    z_train_mean = z_train.mean()
    X_train = X_train-X_train_mean
    X_test = X_test-X_train_mean
    z_train = z_train-z_train_mean

    # calculate mse test and train data    
    beta = ols(X_train, z_train)
    z_pred_test = X_test @ beta 
    z_pred_train = X_train @ beta 
    mse_test[i]=MSE(z_pred_test, z_test)
    mse_train[i]=MSE(z_pred_train, z_train)


fig = plt.figure()
plt.plot(degree, mse_test, label='MSE test')
plt.plot(degree, mse_train, label='MSE train')
#plt.yscale('log')          # test to see how error goes down with increase in degree
#plt.xscale('log')
plt.xlabel('Polynomial degree')
plt.ylabel('Mean squared error (MSE)')
plt.grid()
plt.legend()
fig.savefig('terrain_mse_train_test.png')