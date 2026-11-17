#%%
from funcs import get_samples_smooth, get_chi2_at_point, get_models_z, get_samples_irsic, plot_best_fit_vs_irsic
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

# Fiducial analysis:
#name = 'chain_fwdm_fiducial'
name = 'chain_fwdm_patchy_Rsu0_post_tau'
directory = f'/data/og313/lyalpha/code_wdm/archive_chains/{name}'
burning = 0.8
samples = get_samples_smooth(directory, burning)

if 'patchy' in name:
    base_path = '/data/og313/lyalpha/code_wdm/analysis_patchy'
elif 'emu' in name:
    base_path = '/data/og313/lyalpha/code_wdm/analysis_RC_cdm_u0_emu'
else:
    base_path = '/data/og313/lyalpha/code_wdm/analysis_no_RC'
name_chain = 'chain_bo_WDM_pRELICS_patchy_T0G_mru0_moC'
print('Name of the analysis is', name)

chi42_value, chi46_value, chi5_value, bf_pred_42, bf_pred_46, bf_pred_5 = get_chi2_at_point(name_chain, samples, base_path, post=True, irsic=False, kmax = False, new_alpha=0.5)
