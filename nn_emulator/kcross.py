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
z = 5.0
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

true_test_values, pred_test_values = [], []
k = 10
kf = KFold(n_splits=k, shuffle = True, random_state = 1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

fracerr = []
true_test_values = []
pred_test_values = []

start_time = time.time()
fracerr = []
for fold, (train_index, test_index) in enumerate(kf.split(inputs, outputs)):


    print('Processing fold no:', fold+1)
    print('Train:', train_index, 'Test:', test_index)

    # Check if there is any overlap between train and test indices
    overlap = set(train_index).intersection(set(test_index))
    if overlap:
        print(f'Overlap found between train and test indices in fold {fold+1}: {overlap}')
    else:
        print(f'No overlap between train and test indices in fold {fold+1}')


    np.savetxt(f'{base_path}/z{z}/train_index_{fold}.txt', train_index)
    np.savetxt(f'{base_path}/z{z}/test_index_{fold}.txt', test_index)

    # Re-initialize the model at the start of each fold
#     model = NNInterpolator(
#     input_size=6,
#     output_size=16,
#     hidden=120,
#     learning_rate=0.0001,
#     act=torch.nn.ReLU,
#     patience=500,
#     patience_scheduler=30,
#     base_path=base_path
# )

#     # Split the input and output into training and validation sets
#     train_input, val_input = inputs[train_index], inputs[test_index]
#     train_output, val_output = outputs[train_index], outputs[test_index]

    # print(np.shape(train_input), np.shape(val_input))

    # # Convert the numpy arrays to PyTorch tensors
    # # Convert the numpy arrays to PyTorch tensors and then to float
    # train_input, train_output = torch.from_numpy(train_input).float(), torch.from_numpy(train_output).float()
    # val_input, val_output = torch.from_numpy(val_input).float(), torch.from_numpy(val_output).float()

    # # Create TensorDatasets for the training and validation sets
    # train_data = TensorDataset(train_input, train_output)
    # val_data = TensorDataset(val_input, val_output)

    # # Create DataLoaders for the training and validation sets
    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # train_losses, test_losses, y_true, y_pred, best_epoch = model.train_batch(z, train_loader, val_loader, epochs, fold = fold)
    

#     print(np.shape(y_true), np.shape(y_pred))
#     fracerr.append(np.exp(y_pred)/np.exp(y_true))

# print(np.shape(fracerr))
# mean_fracerr = np.mean(np.array(fracerr), axis=0)
# print(np.shape(mean_fracerr))
# ratio = mean_fracerr
# end_time = time.time()
# print('Total time taken in seconds:', end_time - start_time)
