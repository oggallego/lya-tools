#%%

import sys
sys.path.append('/data/og313/lyalpha/code_wdm/nn_emulator_fwdm')
from train import check_test_distribution
from model import NNInterpolator
import torch

target_redshift = 4.2
base_path = '/data/og313/lyalpha/code_wdm/analysis_no_RC'
fold = 'train'
# Step 2: Train the model
print(f"Training model for redshift {target_redshift}...")
model = NNInterpolator(
    input_size=6,
    output_size=16,
    hidden=120, 
    learning_rate=0.0001,
    act=torch.nn.ReLU,
    patience=400,
    patience_scheduler=30,
    base_path=base_path
)
check_test_distribution(model, target_redshift, base_path, fold)
# %%

# %%
