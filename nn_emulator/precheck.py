#%%

import torch
import numpy as np
import os

import matplotlib.collections as mcoll
import matplotlib.cm as cm
from matplotlib import pyplot as plt, ticker as mticker
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
import time
import argparse

from config import redshift_configs
from data_processing import read_data_nn


def check_param_fwdm(target_redshift, base_path, fwdm, m_wdm):

    """
    Reads input and output data and checks flux PS prediction varying one parameter at a time before neural network training.

    Args:
        target redshift (float): from configuration file
        fwdm (float): fraction of WDM
        m_wdm (float): mass of WDM


    Returns:
        figure: saved in models_test with name: e.g. ratio_all_params_{z}_RC_cdm.txt
    """


    z = target_redshift
    # config = redshift_configs[target_redshift]
    # print(f'Checking arguments: base_path: {config["base_path"]}, in file: {config["in_file"]}, out file: {config["out_file"]}')
    print('Inputs and labels read from', base_path)
    inputs = np.loadtxt(f'{base_path}/z{z}/inputs{z}{base_path.split("analysis")[1]}_all.txt')
    labels = np.loadtxt(f'{base_path}/z{z}/ps{z}{base_path.split("analysis")[1]}_all.txt')

    # inputs, labels = read_data_nn(config["base_path"],
    #         config["in_file"],
    #         config["out_file"])
    # alpha = 1/m_wdm
    # Only if wdm and fwdm are passed as parameters of the function:
    if m_wdm == None:
        alpha = 0
    else:
        alpha = 1/m_wdm
    # Example usage:
    params = ['$T_{0}$', rf'$\gamma$', rf'$\tau_{{eff}}$', rf'$u_{{0}}$', r'$\alpha$',  r'$f_{\mathrm{wdm}}$']

    taueff_boera_z = 1.56 * ((1 + z) / 5.75) ** 4

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
            
            sort_outputs = labels[ref][sort_idx]
            ps_true = []
            T0_input = np.arange(5000,15000,1000)
            len_input = len(T0_input)
            T0vals = np.arange(0,10,1)

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
            print('Shape of inputs in fwdm', np.shape(sort_inputs))
            print(f'taking the ratio wrt to model: {sort_inputs[idx_out]}')
            sort_outputs = labels[ref][sort_idx]

        
        logk = np.arange(-2.2, -0.7, 0.1)
        kval = 10**logk
        cmap = cm.viridis
        norm = plt.Normalize(sort_inputs[:, idx].min(), sort_inputs[:, idx].max()) 

        print('Inputs and labels read from', base_path)
        base_path = '/data/og313/lyalpha/code_wdm/analysis_no_RC'
        inputs = np.loadtxt(f'{base_path}/z{z}/inputs{z}{base_path.split("analysis")[1]}_all.txt')
        labels = np.loadtxt(f'{base_path}/z{z}/ps{z}{base_path.split("analysis")[1]}_all.txt')
        print('Inputs use for CDM from', inputs)
        if z == 4.2:
            ref_cdm = np.where((inputs[:, 0] == 10000) & (inputs[:, 1] == 1.2) & (inputs[:, 2] == 1*taueff_boera_z) & (inputs[:, 3] == 8.1477417) &  (inputs[:, 4] ==  0) & (inputs[:, 5] ==  0))
        if z == 4.6:
            ref_cdm = np.where((inputs[:, 0] == 10000) & (inputs[:, 1] == 1.2) & (inputs[:, 2] == 1*taueff_boera_z) & (inputs[:, 3] == 7.6879470) &  (inputs[:, 4] ==  0) & (inputs[:, 5] ==  0))
        elif z==5.0:
            ref_cdm = np.where((inputs[:, 0] == 10000) & (inputs[:, 1] == 1.2) & (inputs[:, 2] == 1*taueff_boera_z) & (inputs[:, 3] ==  6.3834273) &  (inputs[:, 4] ==  0) & (inputs[:, 5] ==  0))
        
        input_cdm  = inputs[ref_cdm] # Since this will e a single input value
        print('Input cdm', input_cdm)
        output_cdm = labels[ref_cdm]
        # print(np.shape(output_cdm[0]))
        
        fontsize = 15
    

        plt.rcParams.update({
            'font.size': fontsize,       # Change the general font size
            'axes.labelsize': fontsize,  # Font size of the x and y labels
            'xtick.labelsize': fontsize, # Font size of the tick labels
            'ytick.labelsize': fontsize, # Font size of the tick labels
            'legend.fontsize': fontsize, # Font size of the legend
            'figure.titlesize': fontsize # Font size of the figure title
        })


        for i in range((np.shape(sort_outputs)[0])):
            # if param == r'$\alpha$':
            #     ax.plot(logk, sort_outputs[i, :]/output_cdm[0], color=cmap(norm(sort_inputs[:, idx][i]))) 
            if param == r'$f_{\mathrm{wdm}}$' or  r'$\alpha$':
                ax.plot(logk, sort_outputs[i, :]/output_cdm[0], color=cmap(norm(sort_inputs[:, idx][i])))
                # ax.plot(logk, output_cdm[0]/output_cdm[0], color=cmap(norm(sort_inputs[:, idx][i])))
            else:
                ax.plot(logk, sort_outputs[i, :]/sort_outputs[idx_out], color=cmap(norm(sort_inputs[:, idx][i]))) 
                
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
    # out_file_base = config["out_file"].split("ps")[1].replace(".txt", "")
    fig.suptitle(f'Fixed $\\alpha$ = {round(alpha, 2)} and fwdm = {fwdm}, z = {z}', fontsize=20)

    # Create the directory path
    directory_path = f'{base_path}/{cosmology}'

    # Check if the directory exists, if not, create it
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    plt.savefig(f'{directory_path}/ratios_all_params_z{target_redshift}.pdf')
    plt.show()
    # plt.show()
    # if rc == True:  
    #     plt.savefig(f'/data/og313/lyalpha/code_wdm/models/wdm{m_wdm}/ratios_all_params_RC_z{z}_wdm{m_wdm}_withpred.pdf')
    # if rc == False:  
    #     plt.savefig(f'/data/og313/lyalpha/code_wdm/models/wdm{m_wdm}/ratios_all_params_z{z}_wdm{m_wdm}.pdf')
    #     print('Have saved plot without RC')




'Change the following for the part#icular redshfit and cosmology to check:'
# # Now for all cosmologies and all redshifts:
z_values = [4.2]
fractions = [0]
# fractions = [0.5]
m_wdms = [4]

base_path = '/data/og313/lyalpha/code_wdm/analysis_patchy'
# # # Call the function 
# with each combination of arguments
for redshift in z_values:
    for mass in m_wdms:
        for fwdm in fractions:
            check_param_fwdm(redshift, base_path, fwdm, mass)
# %%