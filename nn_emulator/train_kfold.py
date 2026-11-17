import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np 
import os
import re
import ast
import sys

import time
import matplotlib.pyplot as plt
from model import NNInterpolator
import argparse
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


epochs = 1



# For the moment, fix the redshift:
z = 4.2
# Load the data
base_path = '/data/og313/lyalpha/code_wdm/analysis_no_RC_kfold'
train_data = np.load(f'/data/og313/lyalpha/code_wdm/analysis_no_RC_kfold/z{z}/train_data.npz')

# Split into x and y
inputs = train_data['x']
outputs = train_data['y']


X = inputs
y = outputs


# Repeat fiducial training:
learning_rate = 0.0001 # fiducial


# Further tests since training is taking long:
batch_size = 18 # need to choose a batch size that matches the dimension of the remaining data set: 0.91%   
patience = 150

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




start_time = time.time()
fracerr = []

index = int(sys.argv[1])
     
print('Training emulator for fold:', index )
start_time = time.time()

train_index = np.loadtxt(f'{base_path}/z{z}/train_index_{index}.txt').astype(int)
test_index = np.loadtxt(f'{base_path}/z{z}/test_index_{index}.txt').astype(int)
print(type(train_index), type(test_index))

# Re-initialize the model at the start of each fold
model = NNInterpolator(
input_size=6,
output_size=16,
hidden=120,
learning_rate=0.0001,
act=torch.nn.ReLU,
patience=500,
patience_scheduler=30,
base_path=base_path
)

# Split the input and output into training and validation sets
train_input, val_input = inputs[train_index], inputs[test_index]
train_output, val_output = outputs[train_index], outputs[test_index]

print(np.shape(train_input), np.shape(val_input))

# Convert the numpy arrays to PyTorch tensors
# Convert the numpy arrays to PyTorch tensors and then to float
train_input, train_output = torch.from_numpy(train_input).float(), torch.from_numpy(train_output).float()
val_input, val_output = torch.from_numpy(val_input).float(), torch.from_numpy(val_output).float()

# Create TensorDatasets for the training and validation sets
train_data = TensorDataset(train_input, train_output)
val_data = TensorDataset(val_input, val_output)

# Create DataLoaders for the training and validation sets
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

train_losses, test_losses, y_true, y_pred, best_epoch = model.train_batch(z, train_loader, val_loader, epochs, fold = index)


print(np.shape(y_true), np.shape(y_pred))
fracerr.append(np.exp(y_pred)/np.exp(y_true))

mean_fracerr = np.mean(np.array(fracerr), axis=0)
print(np.shape(mean_fracerr))

print('Fractional error for this fold from validation set', np.mean(mean_fracerr, axis=0))

end_time = time.time()
print('Total time taken in seconds:', end_time - start_time)

gs = gridspec.GridSpec(4, 5, width_ratios=[1, 1, 1, 1, 0.05])
gs.update(wspace=0.5)

fig = plt.figure(figsize=(20, 20))
plt.subplots_adjust(hspace=0.4)
plt.rcParams.update({'font.size': 25})

true_values = np.exp(y_true)
mean_predictions = np.exp(y_pred)

ratio = np.exp(mean_predictions) / np.exp(true_values)
mean_ratio = np.mean(ratio, axis=0)
print(mean_ratio)

axs = []
for i in range(4):
    for j in range(4):
        axs.append(fig.add_subplot(gs[i, j]))

# Create a scatter plot with a colormap
for ax, kbin in zip(axs, range(16)):
    min_val = min(true_values[:, kbin].min(), mean_predictions[:, kbin].min())
    max_val = max(true_values[:, kbin].max(), mean_predictions[:, kbin].max())
    hist, xedges, yedges = np.histogram2d(true_values[:, kbin], mean_predictions[:, kbin], bins=50)
    
    # Find bin index for each value in true array and pred array given the bin edges from xedges
    xidx = np.clip(np.digitize(true_values[:, kbin], xedges), 0, hist.shape[0]-1)
    yidx = np.clip(np.digitize(mean_predictions[:, kbin], yedges), 0, hist.shape[1]-1)
    # Use the bin indices to look up the values in the 2D histogram:
    # c is the color parameter used in the scatter plot
    c = hist[xidx, yidx]
    sc = ax.scatter(true_values[:, kbin], mean_predictions[:, kbin], c=c, alpha=0.5)
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)  # Add y=x line
    ax.set_xlabel('True Values', fontsize=17)
    ax.set_ylabel('Predicted Values', fontsize=17)
    ax.set_title(f'$k_{{bin}}$ = {kbin}',  fontsize=17)
    ax.tick_params(axis='both', which='major', labelsize=17)  # Increase tick label size for each subplot

# Create a colorbar in the last column of the gridspec
cax = fig.add_subplot(gs[:, -1])
cbar = fig.colorbar(sc, cax=cax)
cbar.set_label('Number of points', fontsize=17)

plt.tight_layout()
# plt.savefig(f'/data/og313b/lyalpha/ML/dropout/log_true_pred_testset_180_bs24.pdf')
plt.show(block=True)

# Set the model to evaluation mode

logk = np.arange(-2.2, -0.7, 0.1)
# print(np.shape(true_values))
# print(np.shape(mean_predictions))

#Plot the mean of predicted/true versus kbin and shaded area is the 1sigma error
plt.figure(figsize=(10, 6))
plt.plot(logk, mean_ratio)
# plt.fill_between(logk, np.array(mean_ratio) - np.array(final_std_per_bin), np.array(mean_ratio) + np.array(final_std_per_bin), color='gray', alpha=0.5)
plt.xlabel('kbin')
plt.ylabel('Mean of predicted/true')
plt.legend()
plt.savefig('test.pdf')
plt.show()