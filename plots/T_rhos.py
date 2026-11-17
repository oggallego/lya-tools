import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from matplotlib import pyplot as plt, ticker as mticker
import matplotlib.cm as cm
from matplotlib import pyplot as plt, ticker as mticker
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
import re
import ast
import os
import sys
sys.path.append('/data/og313b/lyalpha/models/lambdaCDM/ML/z4.2/')  
from NNInterpolator import NNInterpolator, load_data

sys.path.append('/data/og313b/lyalpha/')
import readlos as RL  

file_path = '/data/og313b/lyalpha/models/lambdaCDM/planck1_20_1024/los/los2048_n5000_z4.200.dat'
data = RL.LoSfile(file_path, ignore_tau= True)
rhoker_H = data.rhoker_H #dimensionless
rhoker_H1 = data.rhoker_H1 # dimensionless
tempker_H1 = data.tempker_H1 # K

pix_size = data.box100/2048
vol_pixel = pix_size**3
vol_pixel

log_temp = np.log10(tempker_H1)
log_rho = np.log10(rhoker_H)

# Calculate the volume-weighted temperature-density values
vol_weighted_temp = log_temp * vol_pixel
vol_weighted_rho = log_rho * vol_pixel

# Sum up the volume-weighted temperature-density values
sum_vol_weighted_temp = np.sum(vol_weighted_temp)
sum_vol_weighted_rho = np.sum(vol_weighted_rho)

# Calculate the total volume
total_vol = np.sum(vol_pixel)

# Calculate the volume-weighted temperature-density distribution
vol_weighted_temp_dist = sum_vol_weighted_temp / total_vol
vol_weighted_rho_dist = sum_vol_weighted_rho / total_vol

# Flatten the arrays
log_temp_flat = log_temp.flatten()
log_rho_flat = log_rho.flatten()
vol_pixel_flat = vol_pixel.flatten()

# Create a boolean array where each element is True if the corresponding element in log_temp_flat is above the threshold, and False otherwise
mask = log_temp_flat < 4.75

# Use the mask to get the values in log_temp_flat that are above the threshold
log_temp_flat= log_temp_flat[mask]

# Use the mask to get rid of the corresponding values in log_rho_flat
log_rho_flat = log_rho_flat[mask]


# # Flatten the arrays
# log_temp_flat = log_temp.flatten()
# log_rho_flat = log_rho.flatten()
# vol_pixel_flat = vol_pixel.flatten()

# Calculate the volume-weighted temperature-density distribution by binning
hist, xedges, yedges = np.histogram2d(log_temp_flat, log_rho_flat, bins=50) #all of them have the same weights

# Calculate the volume-weighted temperature-density distribution
vol_weighted_hist = hist * vol_pixel

# Calculate the bin centers
xcenters = (xedges[:-1] + xedges[1:]) / 2
ycenters = (yedges[:-1] + yedges[1:]) / 2


# # Flatten the arrays
# log_temp_flat = log_temp.flatten()
# log_rho_flat = log_rho.flatten()
# vol_pixel_flat = vol_pixel.flatten()


# # Calculate the volume-weighted temperature-density distribution
vol_weighted_hist = hist * vol_pixel

# Calculate the bin centers
ycenters = (yedges[:-1] + yedges[1:]) / 2
xcenters = (xedges[:-1] + xedges[1:]) / 2

# # Create a grid of bin centers
X, Y = np.meshgrid(ycenters, xcenters )

# Flatten the grid and the volume-weighted histogram
# X_flat = X.flatten()
# Y_flat = Y.flatten()
# vol_weighted_hist_flat = vol_weighted_hist.flatten()


plt.scatter(X, Y, c=np.log10(vol_weighted_hist), cmap='viridis')


plt.colorbar(label='Log10(Volume-weighted counts)')


plt.xlabel(r'Log10(Density)')
plt.ylabel('Log10(Temperature)')


plt.show()
