#%%

from model_loader import get_mixed_models
from predictor import get_nn_prediction
from data_processing import read_data_nn
import os

import matplotlib.collections as mcoll
import matplotlib.cm as cm
from matplotlib import pyplot as plt, ticker as mticker
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
import numpy as np

from config import redshift_configs



def check_param_fwdm_with_pred(target_redshift, fwdm, m_wdm):

    'fwdm allowed for alpha =/= 0 and viceversa, but all these models recover CDM'
    config = redshift_configs[target_redshift]
    z = target_redshift

    base_path = '/data/og313/lyalpha/code_wdm/analysis_no_RC_ratio'

    print('Using base path', base_path) # This is defined in run cobaya script to use the current working directory
    inputs = np.loadtxt(f'{base_path}/z{z}/inputs{z}{base_path.split("analysis")[1]}_all.txt')
    print(f'Inputs are inputsz{base_path.split("analysis")[1]}_all.txt')
    print('Shape of all inputs loaded', np.shape(inputs))
    

    labels = np.loadtxt(f'{base_path}/z{z}/ps{z}{base_path.split("analysis")[1]}_all.txt')
    print('Shape of all inputs loaded', np.shape(labels))
    z = target_redshift
    taueff_boera_z = 1.56 * ((1 + z) / 5.75) ** 4
   
    # Only if wdm and fwdm are passed as parameters of the function:
    if m_wdm == None:
        alpha = 0
    else:
        alpha = 1/m_wdm
    # Example usage:
    params = ['$T_{0}$', rf'$\gamma$', rf'$\tau_{{eff}}$', rf'$u_{{0}}$', r'$\alpha$',  r'$f_{\mathrm{wdm}}$']

    # Load models:
    rc = False
    models, min_in, max_in, taueff= get_mixed_models(target_redshift, base_path)# rc)
    
    print('How many models?', len(models))

    fig, axs = plt.subplots(3, 2, figsize=(20, 15)) 
    for i, param in enumerate(params):
        ax = axs[i//2, i%2]
        if param == '$T_{0}$':

            idx = 0
            idx_out = 5
            if z == 4.2:
                ref = np.where((inputs[:, 1] == 1.2) & (inputs[:, 2] == 1*taueff_boera_z) & (inputs[:, 3] ==  8.1477417) & (inputs[:, 4] ==  alpha)  & (inputs[:, 5] ==  fwdm))
            if z == 4.6:
                ref = np.where((inputs[:, 1] == 1.2) & (inputs[:, 2] == 1*taueff_boera_z) & (inputs[:, 3] ==  7.6879470) & (inputs[:, 4] ==  alpha)& (inputs[:, 5] ==  fwdm))
            elif z == 5:
                ref = np.where((inputs[:, 1] == 1.2) & (inputs[:, 2] == 1*taueff_boera_z) & (inputs[:, 3] ==   6.3834273) & (inputs[:, 4] ==   alpha) & (inputs[:, 5] ==  fwdm))
            sort_idx  = inputs[ref][:,idx].argsort()
            sort_inputs = inputs[ref][sort_idx]
            print('Ratio taken wrt to' ,sort_inputs[idx_out])
            print(np.shape(sort_inputs))
            
            sort_outputs = labels[ref][sort_idx]
            ps_true = []
            T0_input = np.arange(5000,15000,1000)
            len_input = len(T0_input)
            T0vals = np.arange(0,10,1)

            print(np.shape(sort_inputs))
            rc = False
            pred = get_nn_prediction(z, sort_inputs, min_in, max_in, models, rc)
            # pred = pred.reshape(-1, 16)
            print('Original pred shape:', np.shape(pred))
            pred = np.array(pred).reshape(len(T0vals),16)
            # print('Pred shape', np.shape(pred))

        if param == rf'$\gamma$':
            idx = 1
            idx_out = 3
            if z == 4.2:
                ref = np.where((inputs[:, 0] == 10000) & (inputs[:, 2] == 1*taueff_boera_z) & (inputs[:, 3] == 8.1477417) & (inputs[:, 4] ==   alpha) & (inputs[:, 5] ==  fwdm))
            if z == 4.6:
                ref = np.where((inputs[:, 0] == 10000) & (inputs[:, 2] == 1*taueff_boera_z) & (inputs[:, 3] ==  7.6879470) & (inputs[:, 4] ==   alpha)& (inputs[:, 5] ==  fwdm))
            if z == 5:
                ref = np.where((inputs[:, 0] == 10000) & (inputs[:, 2] == 1*taueff_boera_z) & (inputs[:, 3] ==   6.3834273) & (inputs[:, 4] ==   alpha)& (inputs[:, 5] ==  fwdm))

            sort_idx  = inputs[ref][:,idx].argsort()
            sort_inputs = inputs[ref][sort_idx]
            sort_outputs = labels[ref][sort_idx]
            rc = False
            pred = get_nn_prediction(z, sort_inputs, min_in, max_in, models, rc)
            pred = np.array(pred).reshape(len(T0vals),16)
            print('Pred shape', np.shape(pred))

        
            
        if param == rf'$\tau_{{eff}}$':
            idx = 2
            idx_out = 7
            if z == 4.2:
                ref = np.where((inputs[:, 0] == 10000) & (inputs[:, 1] == 1.2) & (inputs[:, 3] == 8.1477417) & (inputs[:, 4] ==   alpha) & (inputs[:, 5] ==  fwdm))
            if z == 4.6:
                ref = np.where((inputs[:, 0] == 10000) & (inputs[:, 1] == 1.2) & (inputs[:, 3] ==  7.6879470) & (inputs[:, 4] ==   alpha) & (inputs[:, 5] ==  fwdm))
            if z == 5:
                ref = np.where((inputs[:, 0] == 10000) & (inputs[:, 1] == 1.2) & (inputs[:, 3] ==   6.3834273) & (inputs[:, 4] ==   alpha) & (inputs[:, 5] ==  fwdm))

            sort_idx  = inputs[ref][:,idx].argsort()
            sort_inputs = inputs[ref][sort_idx]
            sort_outputs = labels[ref][sort_idx]

            taueff_input =  np.arange(0.3, 1.8, 0.1)
            len_input = len(taueff_input)
            rc = False
            pred = get_nn_prediction(z, sort_inputs, min_in, max_in, models, rc)
            pred = np.array(pred).reshape(len_input,16)
            print('Pred shape', np.shape(pred))


        if param == rf'$u_{{0}}$':
            if z == 4.2:
                u0_grid = np.array([4.0239696, 4.6623777, 5.6461662, 6.5922353, 7.1101564, 8.1477417, 9.9236799, 11.614156, 13.262641, 15.009738, 18.176741, 21.106591])
            if z == 4.6:
                u0_grid =  np.array([3.6469614, 4.3163684, 5.364912, 6.3899003, 6.589134, 7.687947, 9.587889, 11.434539, 12.536159, 14.414026, 17.83328, 21.078213])
            if z == 5:    
                u0_grid = np.array([2.4634363, 3.4376965, 4.3868741, 4.7135148, 5.3831631, 6.3834273, 8.1135775, 9.2448624, 9.917967, 12.360053, 15.563067, 18.783147])
            idx = 3
            idx_out = 5
            ref = np.where((inputs[:, 0] == 10000) & (inputs[:, 1] == 1.2) & (inputs[:, 2] == 1*taueff_boera_z)  & (inputs[:, 4] ==  alpha) & (inputs[:, 5] ==  fwdm))
            sort_idx  = inputs[ref][:,idx].argsort()
            sort_inputs = inputs[ref][sort_idx]
            # print(sort_inputs)
            sort_outputs = labels[ref][sort_idx]

            u0vals = np.arange(0,12,1)
            rc = False
            pred = get_nn_prediction(z, sort_inputs, min_in, max_in, models, rc)
            pred = np.array(pred).reshape(len(u0vals),16)
            print('Pred shape', np.shape(pred))
        

        if param == r'$\alpha$':
            idx_out = 0

            idx = 4
            idx_out = 0 # wrt to which we take the ratio, which is 0 since alpha_grid goes from 0 to 1/2
            if z == 4.2:
                ref = np.where((inputs[:, 0] == 10000) & (inputs[:, 1] == 1.2) & (inputs[:, 2] == 1*taueff_boera_z) & (inputs[:, 3] ==  8.1477417) & (inputs[:, 5] ==  fwdm))
            if z == 4.6:
                ref = np.where((inputs[:, 0] == 10000) & (inputs[:, 1] == 1.2) & (inputs[:, 2] == 1*taueff_boera_z) & (inputs[:, 3] ==  7.6879470) & (inputs[:, 5] ==  fwdm))
            elif z == 5:
                ref = np.where((inputs[:, 0] == 10000) & (inputs[:, 1] == 1.2) & (inputs[:, 2] == 1*taueff_boera_z) & (inputs[:, 3] ==   6.3834273)  & (inputs[:, 5] ==  fwdm))

            sort_idx  = inputs[ref][:,idx].argsort()
            sort_inputs = inputs[ref][sort_idx]
            print('Shape of inputs in alpha', np.shape(sort_inputs))
            print(sort_inputs)
            sort_outputs = labels[ref][sort_idx]

            alphavals = 4

            rc = False
            pred = get_nn_prediction(z, sort_inputs, min_in, max_in, models, rc)
            pred = np.array(pred).reshape(np.shape(sort_inputs)[0],16)
            print('Pred shape', np.shape(pred))
        
        if param ==  r'$f_{\mathrm{wdm}}$':
            # if we fix alpha = 0 then we just recover CDM

            idx = 5
            idx_out = -1 # wrt to which we take the ratio, which is 0 since alpha_grid goes from 0 to 1/2
            if z == 4.2:
                ref = np.where((inputs[:, 0] == 10000) & (inputs[:, 1] == 1.2) & (inputs[:, 2] == 1*taueff_boera_z) & (inputs[:, 3] ==  8.1477417) & (inputs[:, 4] ==  alpha))
            if z == 4.6:
                ref = np.where((inputs[:, 0] == 10000) & (inputs[:, 1] == 1.2) & (inputs[:, 2] == 1*taueff_boera_z) & (inputs[:, 3] ==  7.6879470) & (inputs[:, 4] ==   alpha))
            elif z == 5:
                ref = np.where((inputs[:, 0] == 10000) & (inputs[:, 1] == 1.2) & (inputs[:, 2] == 1*taueff_boera_z) & (inputs[:, 3] ==   6.3834273)  & (inputs[:, 4] ==  alpha))

            sort_idx  = inputs[ref][:,idx].argsort()
            sort_inputs = inputs[ref][sort_idx]
            print(np.shape(sort_inputs))
            print('Shape of inputs in fwdm', np.shape(sort_inputs))
            print(f'taking the ratio wrt to model: {sort_inputs[idx_out]}')
            sort_outputs = labels[ref][sort_idx]

            fwdmvals = 4

            rc = False
            pred = get_nn_prediction(z, sort_inputs, min_in, max_in, models, rc)
            pred = np.array(pred).reshape(np.shape(sort_inputs)[0],16)
            print('Pred shape', np.shape(pred))

        
        logk = np.arange(-2.2, -0.7, 0.1)
        kval = 10**logk
        cmap = cm.viridis
        norm = plt.Normalize(sort_inputs[:, idx].min(), sort_inputs[:, idx].max()) 


        
        fontsize = 15
        logk = np.arange(-2.2, -0.7, 0.1)
        kval = 10**logk
        plt.rcParams.update({
            'font.size': fontsize,       # Change the general font size
            'axes.labelsize': fontsize,  # Font size of the x and y labels
            'xtick.labelsize': fontsize, # Font size of the tick labels
            'ytick.labelsize': fontsize, # Font size of the tick labels
            'legend.fontsize': fontsize, # Font size of the legend
            'figure.titlesize': fontsize # Font size of the figure title
        })


        for i in range((np.shape(sort_outputs)[0])):
       
            if param == r'$f_{\mathrm{wdm}}$' or  r'$\alpha$':
                ax.plot(logk, sort_outputs[i, :], color=cmap(norm(sort_inputs[:, idx][i])))
                ax.plot(logk, pred[i, :].flatten(), color = 'tab:orange')
            #     # ax.plot(logk, output_cdm[0]/output_cdm[0], color=cmap(norm(sort_inputs[:, idx][i])))
            else:
                ax.plot(logk, sort_outputs[i, :], color=cmap(norm(sort_inputs[:, idx][i]))) 
                ax.plot(logk, pred[i, :].flatten(), color = 'tab:orange')
                
            ax.set_ylabel('kP(k)$/\pi$')
            ax.set_xlabel('$\mathrm{log}_{10}$ ($k$ $\mathrm{[km^{-1}\,s])}$')
            # ax.set_ylim(0.01,0.8)
            ax.set_xlim(-2.25, -0.65)
            # ax.set_yscale('log')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax= ax, label=f'{param}')
        cbar.ax.yaxis.labelpad = 20
                
    cosmology = f'f{fwdm}_wdm{m_wdm}'

    if fwdm ==1:
        cosmology = f'wdm{m_wdm}'
    if m_wdm == None:
        cosmology = 'cdm'
    if fwdm == 0 and alpha !=0 or fwdm != 0 and alpha == 0:
        cosmology = 'cdm'
    print(cosmology)
    plt.subplots_adjust(wspace=0.2)

    # For intermediate points in the simulation grid:
    # Construct the directory path
    directory_path = os.path.join(base_path, cosmology)

    # Check if the directory exists
    if not os.path.exists(directory_path):
    # Create the directory
    	os.makedirs(directory_path)  
    fig.suptitle(f'Fixed $\\alpha$ = {round(alpha, 2)} and fwdm = {fwdm}, z = {z}', fontsize=20)
    out_file_base = config["out_file"].split("ps")[1].replace(".txt", "")
    plt.savefig(f'{directory_path}/test_ratio_z{out_file_base}.pdf')
    plt.show()


z_values = [4.2]#, 5.0]
fractions = [0.125]#, 0.25, 0.5]
m_wdms = [4]# 3, 4]

# # # Call the function with each combination of arguments
for redshift in z_values:
    for mass in m_wdms:
        for fwdm in fractions:
            check_param_fwdm_with_pred(redshift, fwdm, mass)

# %%
