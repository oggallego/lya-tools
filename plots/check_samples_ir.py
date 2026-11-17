#%%
import sys
sys.path.append('/data/og313/lyalpha/code_wdm/code_plots')
from funcs import get_samples_smooth, get_best_fit, get_models_z,get_samples_irsic, get_bf_sigma_irsic_all, get_sigma_irsic_alpha
import numpy as np
import sys
sys.path.append('/data/og313/lyalpha/code_wdm/mcmc')
from get_boera_data import get_data
import sys
sys.path.append('/data/og313/lyalpha/code_wdm/nn_emulator_fwdm')
from model import NNInterpolator
from predictor import get_nn_prediction_mcmc, get_nn_prediction_post_rs_mcmc
from cobaya_analysis import compute_chi_squared
import pandas as pd
import matplotlib.pyplot as plt
from getdist import loadMCSamples
name_chain = '/data/vault/og313/chains_irsic/chain_bo_WDM_pRELICS_T0G_moC_nofqso'
directory = name_chain

smooth_2D = 0.5
smooth_1D = 0.5
ignore_rows = 0.8

# Orden of names in Vid's saved chains
original_names = ['taueffz42','taueffz46', 'taueffz5', 'T0z42', 'T0z46', 'T0z5',
            'gammaz42', 'gammaz46', 'gammaz5', 
            'u0z42', 'u0z46', 'u0z5',
            'sigma8', 'neff', 'alpha']

# Load the samples without altering parameter names yet
# To run this need .chainparamnames in the same folder
samples_vid = loadMCSamples(directory, settings={
    'ignore_rows': ignore_rows,
    'smooth_scale_2D': smooth_2D,
    'smooth_scale_1D': smooth_1D
})

# # Set the original parameter names
samples_vid.setParamNames(original_names)

names =  [
'T0z42', 'T0z46', 'T0z5',
'gammaz42', 'gammaz46', 'gammaz5',
'taueffz42', 'taueffz46', 'taueffz5',
'u0z42', 'u0z46', 'u0z5',
'alpha'
]


param_ranges = {'T0z42': (0.5, 1.4),
                'T0z46': (0.5, 1.4),
                'T0z5': (0.5, 1.4),
                'gammaz42': (0.9, 1.8),
                'gammaz46': (0.9, 1.8),
                'gammaz5': (0.9, 1.8),
                'taueffz42': (0.31303136056832276, 1.773844376553829),
                'taueffz46': (0.42104312690706497, 2.3859110524733684),
                'taueffz5': (0.554854964068882, 3.1441781297236644),
                'u0z42': (4.0239696, 21.106591),
                'u0z46': (3.6469614, 21.078213),
                'u0z5': (2.4634363, 18.783147),
                'alpha': (0, 1e+16)}
# Set the ranges for the parametersp
samples_vid.setRanges(param_ranges)

print('Gelman-Rubin ratio', samples_vid.getGelmanRubin())
total_samples = samples_vid.samples.shape[0] #ids 1 corresponds to number of parameters
print('Total number of samples is', total_samples)

samples_irsic = samples_vid
    ### more accurately
sigma = 1-0.682689492137086
two_sigma = 1-0.954499736103642
three_sigma = 1-0.997300203936740
four_sigma = 1-0.999936657516334# = 4sigma
five_sigma = 1-0.999999426696856 #= 5sigma

upper_bound_sig = (samples_irsic.confidence('alpha', limfrac=sigma, upper=True))
upper_bound_twosig = (samples_irsic.confidence('alpha', limfrac=two_sigma, upper=True))
print('mwdm <', 1/upper_bound_twosig, 'keV  is excluded at 2 sigman on Irsic+24')


stats = samples_irsic.getMargeStats()
#stats.removeParam('neff')
names =  [
'T0z42', 'T0z46', 'T0z5',
'gammaz42', 'gammaz46', 'gammaz5',
'taueffz42', 'taueffz46', 'taueffz5',
'u0z42', 'u0z46', 'u0z5',
'alpha'
]
original_names = ['taueffz42','taueffz46', 'taueffz5', 'T0z42', 'T0z46', 'T0z5',
            'gammaz42', 'gammaz46', 'gammaz5', 
            'u0z42', 'u0z46', 'u0z5',
            'sigma8', 'neff', 'alpha']# 'fqso']
onesigma_all = []
for i in names:

    lims = stats.parWithName(f'{i}').limits
    for conf, lims in zip(samples_irsic.contours,lims):
        # (conf, lim0.lower, lim0.upper,
        # print(f'Parameter {i}: {conf}, {lims.lower} and {lims.upper}')
        if conf == 0.68:
            onesigma_all.append((lims.lower, lims.upper))
        # [0] T0z42, [1] T0 z46 ...


bestfit = np.zeros(len(original_names))
result = samples_irsic.getLikeStats()
for i in range(len(bestfit)):
    bestfit[i] = result.names[i].bestfit_sample
print('Best fit total array before ordering', bestfit)
bestfit_arr = np.array([bestfit[3], bestfit[4], bestfit[5], bestfit[6], bestfit[7],
bestfit[8], bestfit[0], bestfit[1], bestfit[2], bestfit[9], bestfit[10],
bestfit[11], bestfit[-1]])

# %%
