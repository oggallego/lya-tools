#%%
from funcs import get_samples_smooth, get_best_fit, get_models_z, get_samples_irsic, plot_best_fit_vs_irsic
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
name = 'chain_fwdm_u0T0_T0G_nocut'
name = 'chain_fwdm_noise_not0g'
name = 'chain_fwdm_not0g_Rscdm_kmax'

name = 'noise_runs/chain_fwdm_noise_lognorm_clip'
#name = 'chain_fwdm_repatchy_Rsu0_post_tau'
directory = f'/data/og313/lyalpha/code_wdm/archive_chains/{name}'
burning = 0.8
samples = get_samples_smooth(directory, burning)

if 'repatchy' in name:
    base_path = '/data/og313/lyalpha/code_wdm/analysis_repatchy'
elif 'emu' in name:
    base_path = '/data/og313/lyalpha/code_wdm/analysis_RC_cdm_u0_emu'
else:
    base_path = '/data/og313/lyalpha/code_wdm/analysis_no_RC'
#name_chain = 'chain_bo_WDM_pRELICS_patchy_T0G_mru0_moC'

name_chain = '/data/og313/lyalpha/code_wdm/chains_irsic/chain_bo_WDM_pRELICS_An_moC'
print('Name of the analysis is', name)
#chi42_value, chi46_value, chi5_value, bf_pred_42, bf_pred_46, bf_pred_5 = get_best_fit(name_chain, samples, base_path, post=False,  irsic=False, kmax = False)

chi42_value, chi46_value, chi5_value, bf_pred_42, bf_pred_46, bf_pred_5 = get_best_fit(name_chain, samples, base_path, post=False,  irsic=False, kmax = False, noise=True, use_noise = True)


#Irsic+24:
#name_chain = 'chain_bo_WDM_pRELICS_T0G_moC'
name_chain = '/data/og313/lyalpha/code_wdm/chains_irsic/chain_bo_WDM_pRELICS_An_moC'
#name_chain = '/data/og313/lyalpha/code_wdm/chains_irsic/chain_bo_WDM_pRELICS_patchy_u0T0_mru0_moC'

samples_irsic = get_samples_irsic(name_chain)
#print('Base path used', base_path)
#chi42_value_other, chi46_value_other, chi5_value_other, bf_pred_4#2_other, bf_pred_46_other, bf_pred_5_other = get_best_fit(name_ch#ain, samples_irsic, base_path, post=False, irsic=True, kmax = False)
chi42_value_other, chi46_value_other, chi5_value_other, bf_pred_42_other, bf_pred_46_other, bf_pred_5_other = get_best_fit(name_chain, samples_irsic, base_path, post=False,  irsic=True, kmax = False, noise = True, use_noise = False)


# Get plot:
# save_path = '/data/og313/lyalpha/code_wdm/year2/30_01/Rsu0_post_kmax20_3bins_vs_irsic.pdf'
save_path = f"{name.split('fwdm_')[1]}_vs_irsic_{name_chain.split('pRELICS_')[1]}.pdf"
print('Figure saved at', save_path)
# analysis = '$Patchy + {R_{s}}^{cdm}$ $T_{0}$ prior'
analysis = r'Patchy + $\tau_{\mathrm{CMB}}$ + ${R_{s}}^{cdm}$ ($u_{0}$)'

analysis = ''
#analysis = r'Re-Patchy + $T_{0}$ + $\tau$'

save_path = 'bf_lognorm_clip_irsic_no_noise.pdf'
#analysis = '$k$ > 0.1 s/km by 20%'
plot_best_fit_vs_irsic(bf_pred_42, bf_pred_46, bf_pred_5, chi42_value, chi46_value, chi5_value, 
                       bf_pred_42_other, bf_pred_46_other, bf_pred_5_other, chi42_value_other, chi46_value_other, chi5_value_other, save_path, analysis, kmax=False)

# %%
