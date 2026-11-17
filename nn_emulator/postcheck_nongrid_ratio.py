#%%

from model_loader import get_mixed_models
from predictor import nn_interpolation #' get_nn_prediction, get_nn_prediction_ratio
from data_processing import read_data_nn

import matplotlib.collections as mcoll
import matplotlib.cm as cm
from matplotlib import pyplot as plt, ticker as mticker
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
import numpy as np

from config import redshift_configs

def for_fraction_plot(z, alpha, base_path):

    if isinstance(z, np.ndarray):
        z = z.item()

    # config = redshift_configs[z]
    # base_path = config["base_path"]
    # print('Using base path', base_path) # This is defined in run cobaya script to use the current working directory

    # Define the models:
    rc = False
    models, min_in, max_in, taueff= get_mixed_models(z, base_path)
    taueff_boera_z = 1.56 * ((1 + z) / 5.75) ** 4

    new_input_f =  inputs[(inputs[:, 0] == 10000) & (inputs[:, 1] == 1.2) & (inputs[:, 2] == 1*taueff_boera_z) & (inputs[:, 3] == 8.1477417) & (inputs[:,4]==alpha)]
    #print('Input with alpha = 1 and other post-processed parameters at fiducial values:', new_input_f)

    # If no inputs found, create new inputs with the same fixed values
    if new_input_f.size == 0:
        fixed_values = [10000, 1.2, 1 * taueff_boera_z, 8.1477417, alpha]
        new_input_f = np.array([fixed_values])

        # Ensure new_input_f has 6 columns
    if new_input_f.shape[1] == 5:
        new_input_f = np.insert(new_input_f, 5, 0, axis=1)


    print('Input with alpha = 1 and other post-processed parameters at fiducial values:', new_input_f)
    f_grid = np.linspace(0, 1, 5)
    start = np.min(f_grid) 
    end = np.max(f_grid) 
    step = 0.001
    # step = 0.05

    for i in np.arange(start + step, end + step, step):
        index = np.searchsorted(new_input_f[:, 5], i) # find indices where new rows should be inserted
        values = new_input_f[index - 1, [0, 1,2, 3, 4]].copy()
        new_row = np.insert(values, 5, i)
        new_input_f = np.insert(new_input_f, index, new_row, axis=0)
    print(np.shape(new_input_f))


    # Get Pk for CDM at fiducial values and compare:
    # if z == 4.2:
    #         ref_cdm = np.where((inputs[:, 0] == 10000) & (inputs[:, 1] == 1.2) & (inputs[:, 2] == 1*taueff_boera_z) & (inputs[:, 3] == 8.1477417) &  (inputs[:, 4] ==  0) & (inputs[:, 5] ==  0))
    # # if z == 4.6:
    # #     ref_cdm = np.where((inputs[:, 0] == 10000) & (inputs[:, 1] == 1.2) & (inputs[:, 2] == 1*taueff_boera_z) & (inputs[:, 3] == 7.6879470) &  (inputs[:, 4] ==  0) & (inputs[:, 5] ==  0))
    # # elif z==5.0:
    # #     ref_cdm = np.where((inputs[:, 0] == 10000) & (inputs[:, 1] == 1.2) & (inputs[:, 2] == 1*taueff_boera_z) & (inputs[:, 3] ==  6.3834273) &  (inputs[:, 4] ==  0) & (inputs[:, 5] ==  0))
    
    # input_cdm  = inputs[ref_cdm] # Since this will e a single input value
    # print('Input cdm', input_cdm)

    # Get CDM from emulator as well instead of:
    # output_cdm = labels[ref_cdm]
    

    return new_input_f,  models, min_in, max_in

def plot_for_param(ax, new_input_param, idx_param, avg_preds_param, param, alpha_value):#, avg_pred_cdm):
    logk = np.arange(-2.2, -0.7, 0.1)
    kval = 10**logk

    cmap = cm.viridis
    norm = plt.Normalize(new_input_param[:,idx_param].min(), new_input_param[:,idx_param].max()) 

    # fig, ax = plt.subplots(figsize=(10, 8))
    plt.rcParams.update({
        'font.size': 50,       # Change the general font size
        'axes.labelsize': 50,  # Font size of the x and y labels
        'xtick.labelsize': 100, # Font size of the tick labels
        'ytick.labelsize': 100, # Font size of the tick labels
        'legend.fontsize': 50, # Font size of the legend
        'figure.titlesize': 50 # Font size of the figure title
    })
    # Create a LineCollection object
    for i in range((np.shape(avg_preds_param)[0])):
        # ax.plot(logk, avg_preds_param[i, :]/avg_pred_cdm, color=cmap(norm(new_input_param[:,idx_param][i])))
        ax.plot(logk, avg_preds_param[i, :],color=cmap(norm(new_input_param[:,idx_param][i])))
        ax.set_ylabel('$({{P_{f}}/{P_{ref}}})^{ emulator}$')
        ax.set_xlabel('$\mathrm{log}_{10}$ ($k$ $\mathrm{[km^{-1}\,s])}$')
        # ax.set_ylim(0.01,0.8)
        ax.set_xlim(-2.25, -0.65)
        # ax.set_yscale('log')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        

    # Add a color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax= ax, label=param)
    cbar.ax.yaxis.labelpad = 20 
    mwdm = 1/alpha_value
    plt.title(rf'$m_{{\mathrm{{WDM}}}}$ = {mwdm} keV')

    # plt.show()

def get_nn_prediction(z, new_input, min_in, max_in, models, rc):
    """
    Get predictions for a given input using all models for the specified redshift.

    Args:
        z (float): Redshift value.
        new_input (np.ndarray): Input data for prediction.
        min_in (np.ndarray): Minimum values of input features.
        max_in (np.ndarray): Maximum values of input features.
        models (list): List of trained models for the redshift.

    Returns:
        np.ndarray: Final predictions after applying BC correction.
    """
    # Define a mapping of redshifts to BC correction file paths
    bc_cdm_paths = {
        4.2: "/data/og313b/lyalpha/models/cdm/planck1_20_1024/bc_4.2_1.0.txt",
        4.6: "/data/og313b/lyalpha/models/cdm/planck1_20_1024/bc_4.6_1.0.txt",
        5.0: "/data/og313b/lyalpha/models/cdm/planck1_20_1024/bc_5.0_1.0.txt"
    }
    # Fiducial analysis no RC applies a post correction:L
    rc_cdm_paths = {
        4.2: "/data/og313b/lyalpha/models/cdm/planck1_20_1024/Rs10_4.2_1.0.txt",
        4.6: "/data/og313b/lyalpha/models/cdm/planck1_20_1024/Rs10_4.6_1.0.txt",
        5.0: "/data/og313b/lyalpha/models/cdm/planck1_20_1024/Rs10_5.0_1.0.txt"
    }

    # Ensure the redshift has a corresponding BC correction file
    if z not in bc_cdm_paths:
        raise ValueError(f"No BC correction file path defined for redshift {z}")

    # Get the BC correction file path for the redshift
    bc_cdm_path = bc_cdm_paths[z]
    bc_cdm = np.loadtxt(bc_cdm_path)
    
    rc_cdm_path = rc_cdm_paths[z]
    rc_cdm = np.loadtxt(rc_cdm_path)
    # new_input[0] = new_input[0] * 1e4
    # taueff_boera_z = 1.56 * ((1 + z) / 5.75) ** 4
    # new_input[2] = new_input[2] / taueff_boera_z

    # Predict using all models
    num_models = len(models)
  
    new_out = np.zeros((np.shape(new_input)[0], num_models, 16))
    avg_preds = []
    for i in range(np.shape(new_input)[0]):
        print('Input going into the NN', new_input[i])
        for j in range(num_models):
            new_out[i, j] = nn_interpolation(new_input[i], min_in, max_in, models[j])
    avg_preds = np.mean(new_out, axis=1)

    # Average prediction
    if rc == True:
        print('Inlcuding both box correction and resolution correction!')
        avg_preds_final = avg_preds * bc_cdm * rc_cdm
    elif rc == False:
        print('Not including box correction!')
        avg_preds_final = avg_preds# * bc_cdm
    return avg_preds_final


# Run the two functions:
# Load the data:
z = 4.2

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

#alpha_values = [1, 1/2, 1/3, 1/4, 1/5, 1/6]
# alpha_values = [1/7, 1/8, 1/9, 1/10, 1/11, 1/12]
# alpha_values = np.linspace(1/2, 1, 6)

# Set up the grid of subplots with increased horizontal spacing
# fig, axes = plt.subplots(2, 3, figsize=(50, 30), gridspec_kw={"wspace": 0.6, "hspace": 0.2 })  # Increased wspace
# axes = axes.flatten()  # Flatten to easily iterate over the axes

# z = 4.2
# config = redshift_configs[z]
# base_path = '/data/og313/lyalpha/code_wdm/analysis_no_RC_ratio'
# inputs = np.loadtxt(f'{base_path}/z{z}/inputs{z}{base_path.split("analysis")[1]}_all.txt')
# labels = np.loadtxt(f'{base_path}/z{z}/ps{z}{base_path.split("analysis")[1]}_all.txt')

# # Determine global vmin and vmax for consistent color scale
# vmin, vmax = None, None
# all_avg_preds_f = []

# for alpha_value in alpha_values:
#     new_input_f, models, min_in, max_in = for_fraction_plot(z, alpha_value, base_path)
#     rc = False #Without box correction
#     # avg_preds_f = np.array(get_nn_prediction(z, new_input_f, min_in, max_in, models, rc)).reshape(np.shape(new_input_f)[0], 16)
#     avg_preds_f = np.array(get_nn_prediction(z, new_input_f, min_in, max_in, models, rc)).reshape(np.shape(new_input_f)[0], 16)
#     all_avg_preds_f.append(avg_preds_f)
#     current_min, current_max = avg_preds_f.min(), avg_preds_f.max()
#     if vmin is None or current_min < vmin:
#         vmin = current_min
#     if vmax is None or current_max > vmax:
#         vmax = current_max

# Create a colormap and normalizer for consistent scaling
# norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
# cmap = cm.viridis  # Replace 'viridis' with your desired colormap

# # Create subplots with consistent color scale
# for i, (alpha_value, avg_preds_f) in enumerate(zip(alpha_values, all_avg_preds_f)):
#     new_input_f, models, min_in, max_in = for_fraction_plot(z, alpha_value, base_path)
#     # avg_pred_cdm = np.array(get_nn_prediction(z, input_cdm, min_in, max_in, models)).reshape(-1)

#     # Use the appropriate subplot
#     ax = axes[i]
#     plot_for_param(
#         ax,
#         new_input_f,
#         5,
#         avg_preds_f,
#         r"$f_{wdm}$",
#         alpha_value)#,
#        # avg_pred_cdm,
#     #)
    
#     # Set a title for each subplot
#     mwdm = 1/alpha_value
#     ax.set_title(rf'$m_{{\mathrm{{WDM}}}}$ = {mwdm} keV')
    
#     # Add y-labels to all plots
#     ax.set_ylabel('$({{P_{f}}/{P_{ref}}})^{ emulator}$')
    
#     # Set x-labels only for the bottom row
#     if i // 3 == 1:  # Bottom row (indices 3, 4, 5)
#         ax.set_xlabel('$\mathrm{log}_{10}$ ($k$ $\mathrm{[km^{-1}\,s])}$')
#     else:
#         ax.set_xlabel(None)
#         ax.tick_params(labelbottom=False)

# # Adjust layout and save
# plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for the colorbar
# # plt.savefig('/data/og313/lyalpha/code_wdm/analysis_no_RC_ratio/test_recover_low_alpha.pdf')
# plt.show()
        


# TRY TO PRODUCE THE ANIMATION:
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Define the animate function to create the animation
def animate(i):
    for j, (alpha_value, avg_preds_f) in enumerate(zip(alpha_values, all_avg_preds_f)):
        ax = axes[j]
        ax.clear()  # Clear previous frame
        
        new_input_f, _, _, _ = for_fraction_plot(z, alpha_value, base_path)
        
        # Plot only up to the current frame (i)
        ax.plot(logk, avg_preds_f[:i+1, :].T, color=cmap(norm(new_input_f[:, 5][:i+1])))
        ax.set_title(rf'$m_{{\mathrm{{WDM}}}}$ = {1/alpha_value} keV')
        ax.set_ylabel('$({{P_{f}}/{P_{ref}}})^{ emulator}$')
        
        if j // 3 == 1:  # Bottom row
            ax.set_xlabel('$\mathrm{log}_{10}$ ($k$ $\mathrm{[km^{-1}\,s])}$')
        else:
            ax.set_xlabel(None)
            ax.tick_params(labelbottom=False)

# Setting up the figure and subplots
fig, axes = plt.subplots(2, 3, figsize=(50, 30), gridspec_kw={"wspace": 0.6, "hspace": 0.2})
axes = axes.flatten()

z = 4.2
base_path = '/data/og313/lyalpha/code_wdm/analysis_no_RC_ratio'
inputs = np.loadtxt(f'{base_path}/z{z}/inputs{z}{base_path.split("analysis")[1]}_all.txt')
labels = np.loadtxt(f'{base_path}/z{z}/ps{z}{base_path.split("analysis")[1]}_all.txt')

# Prepare data
alpha_values = [1/7, 1/8, 1/9, 1/10, 1/11, 1/12]
vmin, vmax = None, None
all_avg_preds_f = []
logk = np.arange(-2.2, -0.7, 0.1)
cmap = cm.viridis

for alpha_value in alpha_values:
    new_input_f, models, min_in, max_in = for_fraction_plot(z, alpha_value, base_path)
    avg_preds_f = np.array(get_nn_prediction(z, new_input_f, min_in, max_in, models, False)).reshape(np.shape(new_input_f)[0], 16)
    all_avg_preds_f.append(avg_preds_f)
    
    current_min, current_max = avg_preds_f.min(), avg_preds_f.max()
    if vmin is None or current_min < vmin:
        vmin = current_min
    if vmax is None or current_max > vmax:
        vmax = current_max

# Create a colormap and normalizer for consistent scaling
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

# Set up the animation
ani = animation.FuncAnimation(fig, animate, frames=len(all_avg_preds_f[0]), interval=200, repeat=False)

# Save the animation if needed
ani.save('model_animation.gif', writer='pillow', dpi=200)

plt.show()




# %%
