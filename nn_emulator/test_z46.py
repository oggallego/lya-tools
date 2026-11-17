#%%

from data_processing import load_preprocessed_data
from model import NNInterpolator
from utils.plotting import plot_loss_vs_epoch

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import time

target_redshift = 5.0   
z = target_redshift
base_path = '/data/og313/lyalpha/code_wdm/analysis_no_RC_kfold'
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



base_path = '/data/og313/lyalpha/code_wdm/analysis_no_RC_kfold'
test_data = np.load(f'/data/og313/lyalpha/code_wdm/analysis_no_RC_kfold/z{z}/test_data.npz')

test_x = test_data['x']
test_y = test_data['y']

# Convert to tensors
test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
test_y_tensor = torch.tensor(test_y, dtype=torch.float32)

# Load the model's saved state
fold = 1
model_path = f"{base_path}/z{z}/last_model_foldno_{fold}_120x3.pth"
print('Reading the model in', model_path)
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.model.load_state_dict(state_dict)
model.model.eval()

# Predict on the test set
with torch.no_grad():
    test_predictions = model.model(test_x_tensor)

# Convert predictions and true values to numpy arrays
test_predictions = test_predictions.numpy()
test_output = test_y_tensor.numpy()

# Compute the ratio of predictions to true values
ratio = np.exp(test_predictions) / np.exp(test_output)
median = np.percentile(ratio, 50, axis=0)
lower_68 = np.percentile(ratio, 16, axis=0)
upper_68 = np.percentile(ratio, 84, axis=0)
lower_95 = np.percentile(ratio, 2.5, axis=0)
upper_95 = np.percentile(ratio, 97.5, axis=0)

# Print metrics
print(f"Median: {median}")
print(f"68% confidence interval: {lower_68} - {upper_68}")
print(f"95% confidence interval: {lower_95} - {upper_95}")

one_sigma_error = upper_68 - lower_68
print(f"One sigma error: {one_sigma_error}")


# Generate the plot
logk = np.arange(-2.2, -0.7, 0.1)  # Example range of log(k)
plt.figure(figsize=(8, 6))
plt.rcParams.update({'font.size': 21})

# Median line
plt.plot(logk, median, color='mediumblue', linewidth=2, label="Median")

# Confidence intervals
plt.fill_between(logk, lower_68, upper_68, color='blue', alpha=0.5)#, label="68% CI")
plt.fill_between(logk, lower_95, upper_95, color='cornflowerblue', alpha=0.5)#, label="95% CI")

# Reference lines
plt.axhline(y=1.01, color='dimgray', linestyle='--')
plt.axhline(y=0.99, color='dimgray', linestyle='--')
plt.axhline(y=1.00, color='darkslategray', linestyle='-')

# Formatting
plt.xlim(-2.2, -0.7)
plt.xticks([-2.2, -1.5, -1.0])
plt.ylabel('Predicted/True')
plt.xlabel('$\mathrm{log}_{10}$ ($k$ $\mathrm{[km^{-1}\,s]})$')
plt.legend()
plt.tight_layout()

# Save and show the plot
    # Save the loss plot
output_path = model.get_output_path(z)
print(f'File saved in {output_path}/kcross_train_fold_120x3_test.pdf')
output_file = f"{output_path}/kcross_train_fold_120x3_test.pdf"
plt.savefig(output_file)
plt.show()

# %%
