# Given a base_path save, min_in max_in normalization in file
import numpy as np
import os
import torch
import sys
sys.path.append('/data/og313/lyalpha/code_wdm/nn_emulator_fwdm')
from model import NNInterpolator

z = 5.0
print('For redshift', z)
base_path = '/data/og313/lyalpha/code_wdm/analysis_no_RC'
inputs = np.loadtxt(f'{base_path}/z{z}/inputs{z}{base_path.split("analysis")[1]}_all.txt')
print('Shape of all inputs loaded', np.shape(inputs))

labels = np.loadtxt(f'{base_path}/z{z}/ps{z}{base_path.split("analysis")[1]}_all.txt')
print('Shape of all inputs loaded', np.shape(labels))

learning_rate = 0.0001  # Fiducial value
patience = 30
hidden = 120
act = torch.nn.ReLU
model = NNInterpolator(
input_size=inputs.shape[1],
output_size=labels.shape[1],
hidden=hidden,
learning_rate=learning_rate,
act=act,
patience=patience,
patience_scheduler=30,  # Not needed for inference
base_path=base_path)

models = []

_, _, min_in, max_in = model.preprocess_data(z, inputs, np.log(labels), batch_size=18)

np.savetxt(f'{base_path}/norm_{z}.txt', [min_in, max_in])
