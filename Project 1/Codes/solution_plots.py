import matplotlib.pyplot as plt
import matplotlib as mpl
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

# Make data to fit model
np.random.seed(2018)
x = np.arange(0, 1, 0.05) 
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)
z_ = franke_function(x,y)                    # unraveled
z = np.ravel(z_)

# Make data to plot modelm contains new data point
np.random.seed(1005)
x_plot = np.arange(0, 1, 0.02) 
y_plot = np.arange(0, 1, 0.02)
x_plot, y_plot = np.meshgrid(x_plot,y_plot)
z_plot_ = franke_function(x_plot,y_plot)                    # unraveled
z_plot = np.ravel(z_plot_)


# Best order of polynomial found in project
m = 5                                             

# Best ridge and lasso regularisation found in project
lambd=1e-4

# Design matrix
X = create_design_matrix(x,y,m)
X_plot = create_design_matrix(x_plot,y_plot,m)  # make X_plot to see how solutions behaves when using new data


# Make fig of data used for training model

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
norm = mpl.colors.Normalize(vmin=0, vmax=1.2)
surf = ax.plot_surface(x_plot, y_plot, z_plot_, norm=norm, cmap=cm.coolwarm,
                      linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5) # Color bar 
ax.view_init(azim=-45, elev=30)# Set the azimuth and elevation angles
ax.set_title('Real data surface')
ax.set_zlabel('Elevation')
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlim(0, 1)
plt.savefig('surface_real.png',bbox_inches="tight")

plt.figure()
plt.title('Real data')
plt.imshow(z_plot_, cmap='gray')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('2d_real', bbox_inches="tight")

# Make fig of same data used for training model

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
norm = mpl.colors.Normalize(vmin=0, vmax=1.2)
surf = ax.plot_surface(x, y, z_, norm=norm, cmap=cm.coolwarm,
                      linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5) # Color bar 
ax.view_init(azim=-45, elev=30)# Set the azimuth and elevation angles
plt.title('Real data')
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlim(0, 1)
plt.savefig('surface_real_old_data.png',bbox_inches="tight")

plt.figure()
plt.title('Real data')
plt.imshow(z_, cmap='gray')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('2d_real_old_data', bbox_inches="tight")

# Make fig OLS using new test data
beta_ols = ols(X,z)
z_ols = X_plot @ beta_ols # change to X_plot to see how solutions behaves when using new data
print(z_ols.max())
pred_z_reshape = np.reshape(z_ols, x_plot.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
norm = mpl.colors.Normalize(vmin=0, vmax=1.2)
surf = ax.plot_surface(x_plot, y_plot, pred_z_reshape, norm=norm, cmap=cm.coolwarm,
                      linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5) # Color bar 
ax.view_init(azim=-45, elev=30)# Set the azimuth and elevation angles
plt.title('OLS solution')
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlim(0, 1)
plt.savefig('surface_ols.png',bbox_inches="tight")

plt.figure()
plt.title('OLS solution')
plt.imshow(pred_z_reshape, cmap='gray')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('2d_ols', bbox_inches="tight")

# Make fig OLS using same data
beta_ols = ols(X,z)
z_ols = X @ beta_ols
pred_z_reshape = np.reshape(z_ols, x.shape)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
norm = mpl.colors.Normalize(vmin=0, vmax=1.2)
surf = ax.plot_surface(x, y, pred_z_reshape, norm=norm, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5) # Color bar 
ax.view_init(azim=-45, elev=30)# Set the azimuth and elevation angles
plt.title('OLS surface plotted with train+test data')
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlim(0, 1)
plt.savefig('surface_ols_old_data.png',bbox_inches="tight")

plt.figure()
plt.title('OLS plotted with train+test data')
plt.imshow(pred_z_reshape, cmap='gray')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('2d_ols_old_data', bbox_inches="tight")

# Make fig ridge using new test data
beta_ridge = ridge(X,z,lambd)
z_ridge = X_plot @ beta_ridge   # change to X_plot to see how solutions behaves when using new data
print(z_ridge.max())
pred_z_reshape = np.reshape(z_ridge, x_plot.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
norm = mpl.colors.Normalize(vmin=0, vmax=1.2)
surf = ax.plot_surface(x_plot, y_plot, pred_z_reshape, norm=norm, cmap=cm.coolwarm,
                      linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5) # Color bar 
ax.view_init(azim=-45, elev=30)# Set the azimuth and elevation angles
plt.title('Ridge solution')
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlim(0, 1)
plt.savefig('surface_ridge.png',bbox_inches="tight")

plt.figure()
plt.title('Ridge solution')
plt.imshow(pred_z_reshape, cmap='gray')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('2d_ridge', bbox_inches="tight")

# Make fig ridge using same data
beta_ridge = ridge(X,z,lambd)
z_ridge = X @ beta_ridge
pred_z_reshape = np.reshape(z_ridge, x.shape)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
norm = mpl.colors.Normalize(vmin=0, vmax=1.2)
surf = ax.plot_surface(x, y, pred_z_reshape, norm=norm, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5) # Color bar 
# Set the azimuth and elevation angles
ax.view_init(azim=-45, elev=30)
plt.title('Ridge surface plotted with train+test data')
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlim(0, 1)
plt.savefig('surface_ridge_old_data.png',bbox_inches="tight")

plt.figure()
plt.title('Ridge solution plotted with train+test data')
plt.imshow(pred_z_reshape, cmap='gray')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('2d_ridge_old_data', bbox_inches="tight")

# Make fig lasso using new test data
lassofit = linear_model.Lasso(alpha=lambd,fit_intercept=False, tol=0.0001).fit(X, z)
beta_lasso = lassofit.coef_
z_lasso = X_plot @ beta_lasso
print(z_lasso.max())
pred_z_reshape = np.reshape(z_lasso, x_plot.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
norm = mpl.colors.Normalize(vmin=0, vmax=1.2)
surf = ax.plot_surface(x_plot, y_plot, pred_z_reshape, norm=norm, cmap=cm.coolwarm,
                      linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5) # Color bar 
ax.view_init(azim=-45, elev=30)# Set the azimuth and elevation angles
plt.title('Lasso solution')
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlim(0, 1)
plt.savefig('surface_lasso.png',bbox_inches="tight")

plt.figure()
plt.title('Lasso solution')
plt.imshow(pred_z_reshape, cmap='gray')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('2d_lasso', bbox_inches="tight")

# Make fig ridge using same data
lassofit = linear_model.Lasso(alpha=lambd,fit_intercept=False, tol=0.0001).fit(X, z)
beta_lasso = lassofit.coef_
z_lasso = X @ beta_lasso
pred_z_reshape = np.reshape(z_lasso, x.shape)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
norm = mpl.colors.Normalize(vmin=0, vmax=1.2)
surf = ax.plot_surface(x, y, pred_z_reshape, norm=norm, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5) # Color bar 
# Set the azimuth and elevation angles
ax.view_init(azim=-45, elev=30)
plt.title('Lasso surface plotted with train+test data')
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlim(0, 1)
plt.savefig('surface_lasso_old_data.png',bbox_inches="tight")

plt.figure()
plt.title('Lasso solution plotted with train+test data')
plt.imshow(pred_z_reshape, cmap='gray')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('2d_lasso_old_data', bbox_inches="tight")

########################################################################
# Terrain data

# # Make data
terrain = imageio.imread('SRTM_data_Norway_1.tif')
z_squared = terrain[1800:3600, 0:1800]             # take out a square of the data
z_small = z_squared[::10, ::10]                    # use every tenth pixel

#Scaling - to make dataset smaller and make z,x,y on same scale
xydist = 2*55e3 # meters. approx. distance between 8-9 W degree at 60 N

x = np. linspace(0, 1,z_small.shape[1])
y = np. linspace(0, 1,z_small.shape[0])
x, y = np.meshgrid(x,y)
z_small_mesh=z_small* (2/xydist)
z = np.ravel(z_small_mesh)
print(z.min(), z.max())

z_small_plot = z_squared[::8, ::8]
x_plot = np. linspace(0, 1,z_small_plot.shape[1])
y_plot = np. linspace(0, 1,z_small_plot.shape[0])
x_plot, y_plot = np.meshgrid(x_plot,y_plot)
z_small_plot_mesh=z_small_plot* (2/xydist)
z_plot = np.ravel(z_small_plot_mesh)


# Best order of polynomial found in project
m = 25                                             

# Best ridge and lasso regularisation found in project
lambd=1e-7

# Design matrix
X = create_design_matrix(x,y,m)
X_plot = create_design_matrix(x_plot,y_plot,m)  # make X_plot to see how solutions behaves when using new data


# Make fig of new data

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
norm = mpl.colors.Normalize(vmin=0, vmax=0.020)
surf = ax.plot_surface(x_plot, y_plot, z_small_plot_mesh, norm=norm, cmap=cm.coolwarm,
                      linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5) # Color bar 
ax.view_init(azim=45, elev=30)# Set the azimuth and elevation angles
plt.title('Real data surface')
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('z')
ax.set_zlim(0, 0.025)
plt.savefig('terrain_surface_real.png',bbox_inches="tight")

plt.figure()
plt.title('Real data')
plt.imshow(z_small_plot_mesh, cmap='gray')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('terrain_2d_real', bbox_inches="tight")

# Make fig of same data used for training model

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
norm = mpl.colors.Normalize(vmin=0, vmax=0.02)
surf = ax.plot_surface(x, y, z_small_mesh, norm=norm, cmap=cm.coolwarm,
                      linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5) # Color bar 
ax.view_init(azim=45, elev=30)# Set the azimuth and elevation angles
plt.title('Real data')
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlim(0, 0.025)
plt.savefig('terrain_surface_real_old_data.png',bbox_inches="tight")

plt.figure()
plt.title('Real data')
plt.imshow(z_small_mesh, cmap='gray')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('terrain_2d_real_old_data', bbox_inches="tight")

# Make fig OLS using new test data
beta_ols = ols(X,z)
z_ols = X_plot @ beta_ols # change to X_plot to see how solutions behaves when using new data
print(z_ols.max())
pred_z_reshape = np.reshape(z_ols, x_plot.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
norm = mpl.colors.Normalize(vmin=0, vmax=0.02)
surf = ax.plot_surface(x_plot, y_plot, pred_z_reshape, norm=norm, cmap=cm.coolwarm,
                      linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5) # Color bar 
ax.view_init(azim=45, elev=30)# Set the azimuth and elevation angles
plt.title('OLS solution')
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlim(0, 0.025)
plt.savefig('terrain_surface_ols.png',bbox_inches="tight")

plt.figure()
plt.title('OLS solution')
plt.imshow(pred_z_reshape, cmap='gray')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('terrain_2d_ols', bbox_inches="tight")

# Make fig OLS using same data
beta_ols = ols(X,z)
z_ols = X @ beta_ols
pred_z_reshape = np.reshape(z_ols, x.shape)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
norm = mpl.colors.Normalize(vmin=0, vmax=0.02)
surf = ax.plot_surface(x, y, pred_z_reshape, norm=norm, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5) # Color bar 
ax.view_init(azim=45, elev=30)# Set the azimuth and elevation angles
plt.title('OLS surface plotted with train+test data')
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlim(0, 0.025)
plt.savefig('terrain_surface_ols_old_data.png',bbox_inches="tight")

plt.figure()
plt.title('OLS plotted with train+test data')
plt.imshow(pred_z_reshape, cmap='gray')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('terrain_2d_ols_old_data', bbox_inches="tight")

# Make fig ridge using new test data
beta_ridge = ridge(X,z,lambd)
z_ridge = X_plot @ beta_ridge   # change to X_plot to see how solutions behaves when using new data
print(z_ridge.max())
pred_z_reshape = np.reshape(z_ridge, x_plot.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
norm = mpl.colors.Normalize(vmin=0, vmax=0.02)
surf = ax.plot_surface(x_plot, y_plot, pred_z_reshape, norm=norm, cmap=cm.coolwarm,
                      linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5) # Color bar 
ax.view_init(azim=45, elev=30)# Set the azimuth and elevation angles
plt.title('Ridge solution')
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlim(0, 0.025)
plt.savefig('terrain_surface_ridge.png',bbox_inches="tight")

plt.figure()
plt.title('Ridge solution')
plt.imshow(pred_z_reshape, cmap='gray')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('terrain_2d_ridge', bbox_inches="tight")

# Make fig ridge using same data
beta_ridge = ridge(X,z,lambd)
z_ridge = X @ beta_ridge
pred_z_reshape = np.reshape(z_ridge, x.shape)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
norm = mpl.colors.Normalize(vmin=0, vmax=0.02)
surf = ax.plot_surface(x, y, pred_z_reshape, norm=norm, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5) # Color bar 
# Set the azimuth and elevation angles
ax.view_init(azim=45, elev=30)
plt.title('Ridge surface plotted with train+test data')
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlim(0, 0.025)
plt.savefig('terrain_surface_ridge_old_data.png',bbox_inches="tight")

plt.figure()
plt.title('Ridge solution plotted with train+test data')
plt.imshow(pred_z_reshape, cmap='gray')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('terrain_2d_ridge_old_data', bbox_inches="tight")

# Make fig lasso using new test data
lassofit = linear_model.Lasso(alpha=lambd,fit_intercept=False, tol=0.0001).fit(X, z)
beta_lasso = lassofit.coef_
z_lasso = X_plot @ beta_lasso
print(z_lasso.max())
pred_z_reshape = np.reshape(z_lasso, x_plot.shape)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
norm = mpl.colors.Normalize(vmin=0, vmax=0.02)
surf = ax.plot_surface(x_plot, y_plot, pred_z_reshape, norm=norm, cmap=cm.coolwarm,
                      linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5) # Color bar 
ax.view_init(azim=45, elev=30)# Set the azimuth and elevation angles
plt.title('Lasso solution')
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlim(0, 0.025)
plt.savefig('terrain_surface_lasso.png',bbox_inches="tight")

plt.figure()
plt.title('Lasso solution')
plt.imshow(pred_z_reshape, cmap='gray')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('terrain_2d_lasso', bbox_inches="tight")

# Make fig ridge using same data
lassofit = linear_model.Lasso(alpha=lambd,fit_intercept=False, tol=0.0001).fit(X, z)
beta_lasso = lassofit.coef_
z_lasso = X @ beta_lasso
pred_z_reshape = np.reshape(z_lasso, x.shape)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
norm = mpl.colors.Normalize(vmin=0, vmax=0.02)
surf = ax.plot_surface(x, y, pred_z_reshape, norm=norm, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5) # Color bar 
# Set the azimuth and elevation angles
ax.view_init(azim=45, elev=30)
plt.title('Lasso surface plotted with train+test data')
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlim(0, 0.025)
plt.savefig('terrain_surface_lasso_old_data.png',bbox_inches="tight")

plt.figure()
plt.title('Lasso solution plotted with train+test data')
plt.imshow(pred_z_reshape, cmap='gray')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('terrain_2d_lasso_old_data', bbox_inches="tight")