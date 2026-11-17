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
# import sys
# sys.path.append('/data/og313b/lyalpha/models/lambdaCDM/ML/z4.2/')  
# from NNInterpolator import NNInterpolator, load_data
import sys
sys.path.append('/data/og313b/lyalpha/')
import readlos as RL  

logk = np.arange(-2.2, -0.7, 0.1)

# Use get_ps_quick to get PS at original T0, gamma, taueff = 1.0
sys.path.append('/data/og313/lyalpha/git/get_tau')
import compute_absorption
from get_ps_quick import get_ps_quick

wdm_cosmologies = ['wdm4', 'wdm3', 'wdm2','wdm1']
scaling_factors = np.arange(0.3, 1.8, 0.1)
no_bin = False


def get_ps_for_plot(cosmology, z):
    if cosmology == 'cdm':
        print('Base path')
        base_path = f'/data/og313b/lyalpha/models/cdm/planck1_20_1024/los/'
        print(base_path)
        sim = f'planck1_20_1024'
    if cosmology == 'cdm_new':
        base_path = f'/data/og313b/lyalpha/models/cdm_new/planck1_20_1024/los/'
        sim = f'planck1_20_1024'
    if 'cdm' not in cosmology: 
        sim = f'planck1_20_1024_{cosmology}'
        base_path = f'/data/og313b/lyalpha/models/{cosmology}/planck1_20_1024_{cosmology}/los/'
    print('No error until here')
    print(base_path)
    ps= get_ps_quick(cosmology, sim, z, scaling_factors, base_path, 1.0, no_bin = False)
    ps, freq= ps.get_ps()
    return ps 


z = [4.2]
ps_all_wdm = []
for i in wdm_cosmologies:
    ps_all_wdm.append(get_ps_for_plot(i, z))

z = [4.2]
ps_cdm = get_ps_for_plot('cdm', z)

# Need CDM with CLASS (new)
z = [4.2]
ps_cdm_new = get_ps_for_plot('cdm_new', z)


def get_ps_for_plot_fwdm(fwdm, z):

    sim = f'planck1_20_1024_fwdm{fwdm}_mwdm2'
    cosmology = 'f{fwdm}_wdm2'
    base_path = f'/data/og313b/lyalpha/models/f{fwdm}_wdm2/planck1_20_1024_fwdm{fwdm}_mwdm2.0/los/'
    ps= get_ps_quick(cosmology, sim, z, scaling_factors, base_path, 1.0, no_bin = False)
    ps, freq= ps.get_ps()
    return ps 


ps_all_fwdm = []
fwdms = [0.5, 0.25, 0.125]
z = [4.2]
for i in fwdms:
    ps_all_fwdm.append(get_ps_for_plot_fwdm(i, z))


for i in range(4):
  print(np.array(np.abs(ps_all_wdm[i][z])))
  

import numpy as np

# Load the data from the specified files
covariance_matrix = np.loadtxt('/data/og313/lyalpha/code_wdm/to_csd3/boera_data/covariance_boera.txt')
P_kRC42_new = np.loadtxt('/data/og313/lyalpha/code_wdm/to_csd3/boera_data/P_kRC42_boera.txt')
P_kRC46_new = np.loadtxt('/data/og313/lyalpha/code_wdm/to_csd3/boera_data/P_kRC46_boera.txt')
P_kRC5_new = np.loadtxt('/data/og313/lyalpha/code_wdm/to_csd3/boera_data/P_kRC5_boera.txt')


###PLOT

### Recompute figure to check new WDM1 los files are correct:
import matplotlib.pyplot as plt

z = 4.2


# Update the font size
plt.rcParams.update({'font.size': 27})

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(20, 10))

# Add sigma values:
sigma_values = np.diagonal(np.sqrt(covariance_matrix[:16, :16])) / np.array(ps_cdm[z])


# Plot something on the first subplot
#labels1 = [r'$m_{\mathrm{WDM}} = 1.0\,$keV', r'$m_{\mathrm{WDM}} = 2.0\,$keV', r'$m_{\mathrm{WDM}} = 3.0\,$keV', r'$m_{\mathrm{WDM}} = 4.0\,$keV']
labels1 = [r'$m_{\mathrm{WDM}} = 4.0\,$keV', r'$m_{\mathrm{WDM}} = 3.0\,$keV', r'$m_{\mathrm{WDM}} = 2.0\,$keV', r'$m_{\mathrm{WDM}} = 1.0\,$keV']
lw = 3
for i in range(4):
    ratio = np.array(ps_all_wdm[i][z]) / np.array(ps_cdm[z])
    axs[0].plot(logk, np.array(ps_all_wdm[i][z])/np.array(ps_cdm[z]), label=labels1[i], linewidth=lw)
    #axs[0].fill_between(logk, ratio - sigma_values, ratio + sigma_values, alpha=0.3)
    axs[0].axhline(1, color = 'black', linewidth = lw)
    axs[0].legend()
    axs[0].set_ylabel(r'$P_{F}\,/\,P_{ref}$')
    axs[0].set_xlabel('$\mathrm{log}_{10}$ ($k$ $\mathrm{[km^{-1}\,s])}$')
    axs[0].set_xlim(-2.2, -0.7)
    axs[0].set_ylim(0.2, 1.4)
    axs[0].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    axs[0].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
#axs[0].fill_between(logk, 1 - sigma_values, 1 + sigma_values, alpha=0.3)
# axs[0].bar(logk, sigma_values*2, bottom=1 - sigma_values, color='plum', width = 0.1, alpha=0.5)# label=r'$\pm \sigma$')
# Create two separate legends
# Create two separate legends
handles, labels = axs[0].get_legend_handles_labels()
legend1 = axs[0].legend(handles[:2], labels[:2], loc='upper right', frameon=False)
legend2 = axs[0].legend(handles[2:], labels[2:], loc='lower left', frameon=False)
axs[0].add_artist(legend1)

labels2 = [r'$F_{\mathrm{WDM}} = 0.125$', r'$F_{\mathrm{WDM}} = 0.25$', r'$F_{\mathrm{WDM}} = 0.5$']
linestyles = ['-.', '--', ':']
for i in range(3):
    axs[1].plot(logk, np.array(np.abs((ps_all_fwdm[i][z])))/np.array(np.abs((ps_cdm_new[z]))), label = labels2[i], linestyle = linestyles[i], color = 'tab:green', linewidth=lw)
axs[1].plot(logk, np.array(ps_all_wdm[2][z])/np.array(ps_cdm[z]), label = r'$F_{\mathrm{WDM}} = 1.0$', color  = 'tab:green', linewidth=lw) #No correction factor for WDM
axs[1].axhline(1, color = 'black', linewidth = lw)
axs[1].legend()
axs[1].set_xlabel('$\mathrm{log}_{10}$ ($k$ $\mathrm{[km^{-1}\,s])}$')
axs[1].set_ylim(0.2, 1.4)
axs[1].set_xlim(-2.2, -0.7)
#axs[1].set_yticks(np.arange(0.8, 1.05, 0.05))
axs[1].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
axs[1].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

# sigma_values = np.diagonal(np.sqrt(covariance_matrix[:16, :16])) / np.array(ps_cdm_new[z])
# axs[1].bar(logk, sigma_values*2, bottom=1 - sigma_values, color='plum', width = 0.1, alpha=0.5, label=r'$\pm \sigma$ Boera+19')
axs[1].legend(frameon=False)
plt.subplots_adjust(wspace=0.2)

plt.tight_layout()
plt.savefig('/data/vault/og313/lyalpha/plots_paper/fwdm_mwdm_psz4.2_originalT0gamma_1.0taueff.pdf')
# Show the figure
print(np.max(np.array(ps_all_wdm[-1][z])/np.array(ps_cdm[z])))
plt.show()