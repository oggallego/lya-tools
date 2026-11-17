#%%

from model_loader import get_mixed_models
from predictor import get_nn_prediction, get_nn_prediction_post_rs
from data_processing import read_data_nn
import pandas as pd
import matplotlib.collections as mcoll
import matplotlib.cm as cm
from matplotlib import pyplot as plt, ticker as mticker
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
import numpy as np

def for_fraction_plot(z, alpha, base_path):

    if isinstance(z, np.ndarray):
        z = z.item()

    # config = redshift_configs[z]
    #base_path = config["base_path"]

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
    step = 0.01
    # step = 0.05

    for i in np.arange(start + step, end + step, step):
        index = np.searchsorted(new_input_f[:, 5], i) # find indices where new rows should be inserted
        values = new_input_f[index - 1, [0, 1,2, 3, 4]].copy()
        new_row = np.insert(values, 5, i)
        new_input_f = np.insert(new_input_f, index, new_row, axis=0)
    print(np.shape(new_input_f))


    # Get Pk for CDM at fiducial values and compare:
    if z == 4.2:
        ref_cdm = np.where((inputs[:, 0] == 10000) & (inputs[:, 1] == 1.2) & (inputs[:, 2] == 1*taueff_boera_z) & (inputs[:, 3] == 8.1477417) &  (inputs[:, 4] ==  0) & (inputs[:, 5] ==  0))
    if z == 4.6:
        ref_cdm = np.where((inputs[:, 0] == 10000) & (inputs[:, 1] == 1.2) & (inputs[:, 2] == 1*taueff_boera_z) & (inputs[:, 3] == 7.6879470) &  (inputs[:, 4] ==  0) & (inputs[:, 5] ==  0))
    elif z==5.0:
        ref_cdm = np.where((inputs[:, 0] == 10000) & (inputs[:, 1] == 1.2) & (inputs[:, 2] == 1*taueff_boera_z) & (inputs[:, 3] ==  6.3834273) &  (inputs[:, 4] ==  0) & (inputs[:, 5] ==  0))
    
    input_cdm  = inputs[ref_cdm] # Since this will e a single input value
    print('Input cdm', input_cdm)

    # Get CDM from emulator as well instead of:
    # output_cdm = labels[ref_cdm]
    

    return new_input_f, input_cdm, models, min_in, max_in

def plot_for_param(ax, new_input_param, idx_param, avg_preds_param, param, alpha_value, avg_pred_cdm):
    logk = np.arange(-2.2, -0.7, 0.1)
    kval = 10**logk

    cmap = cm.viridis
    norm = plt.Normalize(new_input_param[:,idx_param].min(), new_input_param[:,idx_param].max()) 

    # fig, ax = plt.subplots(figsize=(10, 8))
    plt.rcParams.update({
        'font.size': 50,       # Change the general font size
        'axes.labelsize': 50,  # Font size of the x and y labels
        'xtick.labelsize': 50, # Font size of the tick labels
        'ytick.labelsize': 50, # Font size of the tick labels
        'legend.fontsize': 50, # Font size of the legend
        'figure.titlesize': 50 # Font size of the figure title
    })
    # Create a LineCollection object
    for i in range((np.shape(avg_preds_param)[0])):
        ax.plot(logk, avg_preds_param[i, :]/avg_pred_cdm, color=cmap(norm(new_input_param[:,idx_param][i])))
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

# Run the two functions:
    # Load the data:
z = 4.2


# config = redshift_configs[z]
# base_path = config["base_path"]
# print('Using base path', base_path) # This is defined in run cobaya script to use the current working directory
# inputs = np.loadtxt(f'{base_path}/z{z}/inputs{z}{base_path.split("analysis")[1]}_all.txt')
# print(f'Inputs are inputsz{base_path.split("analysis")[1]}_all.txt')
# print('Shape of all inputs loaded', np.shape(inputs))
# labels = np.loadtxt(f'{base_path}/z{z}/ps{z}{base_path.split("analysis")[1]}_all.txt')
# print('Shape of all inputs loaded', np.shape(labels))


# alpha_value = 1/3
# new_input_f, input_cdm, models, min_in, max_in = for_fraction_plot(z, alpha_value)

# # Define the models:
# # rc = False
# # models, min_in, max_in, taueff= get_mixed_models(z, config["base_path"])

# avg_preds_f = np.array(get_nn_prediction(z, new_input_f,  min_in, max_in, models)).reshape(np.shape(new_input_f)[0], 16)
# avg_pred_cdm = np.array(get_nn_prediction(z, input_cdm,  min_in, max_in, models)).reshape(-1)

# fig, ax = plt.subplots(figsize=(14, 12))
# plot_for_param(ax, new_input_f, 5, avg_preds_f, r'$f_{wdm}$', alpha_value, avg_pred_cdm )
# plt.savefig(f'/data/og313/lyalpha/code_wdm/year2/22_01/emu_fwdm_check_alpha{alpha_value}.pdf')



import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

alpha_values = [1, 1/2, 1/3, 1/4, 1/5, 1/6]
# alpha_values = [1/7, 1/8, 1/9, 1/10, 1/11, 1/12]
# alpha_values = np.linspace(1/2, 1, 6)

# Set up the grid of subplots with increased horizontal spacing
fig, axes = plt.subplots(2, 3, figsize=(50, 30), gridspec_kw={"wspace": 0.6, "hspace": 0.2 })  # Increased wspace
axes = axes.flatten()  # Flatten to easily iterate over the axes

z = 4.2
# base_path = '/data/og313/lyalpha/code_wdm/analysis_RC_cdm_u0'
base_path = '/data/og313/lyalpha/code_wdm/analysis_patchy'
inputs = np.loadtxt(f'{base_path}/z{z}/inputs{z}{base_path.split("analysis")[1]}_all.txt')
labels = np.loadtxt(f'{base_path}/z{z}/ps{z}{base_path.split("analysis")[1]}_all.txt')


# What does post u0 interpolated Rs correction introduce?
# Define column names
column_names = ['Model', 'zr', 'tau_e', 'u0_4.2<=z<=12.0', 'u0_4.6<=z<=13.0', 'u0_6.0<=z<=13.0']

# Load data from text file, skipping the first 5 rows
u0all = pd.read_csv('/data/og313b/lyalpha/taue_u0_table.txt', skiprows=4, delim_whitespace=True, names=column_names)
models = ['planck1_20_1024', 'planck1_20_1024_cold', 'planck1_20_1024_hot', 'planck1_20_1024_zr525_cold', 'planck1_20_1\
024_zr525', 'planck1_20_1024_zr675', 'planck1_20_1024_zr750']

# Reduce table to models for which we have u0 in wdm cosmlogies:
filtered_u0all = u0all[u0all['Model'].isin(models)].sort_values('u0_4.2<=z<=12.0').reset_index(drop=True)

# Models in increasing u0 values for each redshift:
u042 = filtered_u0all.iloc[:,3].reset_index(drop=True)
u046 = filtered_u0all.iloc[:,4].reset_index(drop=True)
u05 = filtered_u0all.iloc[:,5].reset_index(drop=True)


Rs10_42 = np.load('/data/og313/lyalpha/code_wdm/analysis_RC_cdm_u0_post/Rs10_42.npy')
Rs10_46 = np.load('/data/og313/lyalpha/code_wdm/analysis_RC_cdm_u0_post/Rs10_46.npy')
Rs10_5 = np.load('/data/og313/lyalpha/code_wdm/analysis_RC_cdm_u0_post/Rs10_5.npy')

# Determine global vmin and vmax for consistent color scale
vmin, vmax = None, None
all_avg_preds_f = []

for alpha_value in alpha_values:
    new_input_f, input_cdm, models, min_in, max_in = for_fraction_plot(z, alpha_value, base_path)
    # rc = False
    avg_preds_f = np.array(get_nn_prediction(z, new_input_f, min_in, max_in, models, rc)).reshape(np.shape(new_input_f)[0], 16)
    avg_preds_f = np.array(get_nn_prediction_post_rs(z, new_input_f, min_in, max_in, models, u042, u046, u05, Rs10_42, Rs10_46, Rs10_5)).reshape(np.shape(new_input_f)[0], 16)
    all_avg_preds_f.append(avg_preds_f)
    current_min, current_max = avg_preds_f.min(), avg_preds_f.max()
    if vmin is None or current_min < vmin:
        vmin = current_min
    if vmax is None or current_max > vmax:
        vmax = current_max
print('Shape of input is', np.shape(new_input_f))
# Create a colormap and normalizer for consistent scaling
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap = cm.viridis  # Replace 'viridis' with your desired colormap

# Create subplots with consistent color scale
for i, (alpha_value, avg_preds_f) in enumerate(zip(alpha_values, all_avg_preds_f)):
    new_input_f, input_cdm, models, min_in, max_in = for_fraction_plot(z, alpha_value, base_path)
    
    #Rscdm interpolation:
    rc = False
    # avg_pred_cdm = np.array(get_nn_prediction(z, input_cdm, min_in, max_in, models, rc)).reshape(-1)
    avg_pred_cdm = np.array(get_nn_prediction_post_rs(z, input_cdm, min_in, max_in, models, u042, u046, u05, Rs10_42, Rs10_46, Rs10_5)).reshape(-1)

    # Use the appropriate subplot
    ax = axes[i]
    plot_for_param(
        ax,
        new_input_f,
        5,
        avg_preds_f,
        r"$f_{wdm}$",
        alpha_value,
        avg_pred_cdm,
    )
    
    # Set a title for each subplot
    mwdm = 1/alpha_value
    ax.set_title(rf'$m_{{\mathrm{{WDM}}}}$ = {mwdm} keV')
    
    # Add y-labels to all plots
    ax.set_ylabel('$({{P_{f}}/{P_{ref}}})^{ emulator}$')
    
    # Set x-labels only for the bottom row
    if i // 3 == 1:  # Bottom row (indices 3, 4, 5)
        ax.set_xlabel('$\mathrm{log}_{10}$ ($k$ $\mathrm{[km^{-1}\,s])}$')
    else:
        ax.set_xlabel(None)
        ax.tick_params(labelbottom=False)

# Adjust layout and save
plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for the colorbar
# plt.savefig('/data/og313/lyalpha/code_wdm/year2/24_01/ratio_Pk_fwdm_z42_other_higher.pdf')
# plt.savefig('/data/og313/lyalpha/code_wdm/analysis_no_RC/test_recover_Rscdm_u0_lowalpha.pdf')
plt.show()






# %%

# %%

# %%
