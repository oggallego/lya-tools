#%%
from funcs import get_samples_smooth, get_best_fit, get_models_z, get_samples_irsic, plot_best_fit
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
name = 'chain_fwdm_fiducial'
#name = 'chain_fwdm_u0T0_T0G_nocut'
#name = 'chain_fwdm_t0g_tau'

# k max cuts:
#name = 'chain_fwdm_t0g_Rscdm_kmax'
#name = 'chain_fwdm_not0g_Rscdm_kmax'
#name = 'noise_runs/chain_fwdm_noise_lognorm_clip2'

#Rscdm(u0):
#name = 'chain_fwdm_fiducial_Rsu0_post_tau'
#name = 'chain_fwdm_u0t0_t0g_lowcut_Rsu0_post'

#Repatchy:
#name = 'chain_fwdm_repatchy_Rsu0_post_tau'
#name = 'chain_fwdm_repatchy_u0t0_t0g_Rsu0_post_lowcut'


directory = f'/data/og313/lyalpha/code_wdm/archive_chains/{name}'
burning = 0.8 #0.3 for repatchy Rsu0post u0t0 and tau
samples = get_samples_smooth(directory, burning)

if 'repatchy' in name:
    base_path = '/data/og313/lyalpha/code_wdm/analysis_repatchy'
if 'patchy' in name:
    base_path = '/data/og313/lyalpha/code_wdm/analysis_patchy'
elif 'emu' in name:
    base_path = '/data/og313/lyalpha/code_wdm/analysis_RC_cdm_u0_emu'
else:
    base_path = '/data/og313/lyalpha/code_wdm/analysis_no_RC'
#name_chain = 'chain_bo_WDM_pRELICS_patchy_T0G_mru0_moC'

name_chain = '/data/og313/lyalpha/code_wdm/chains_irsic/chain_bo_WDM_pRELICS_T0G_moC_nofqso'
print('Name of the analysis is', name)
#chi42_value, chi46_value, chi5_value, bf_pred_42, bf_pred_46, bf_pred_5 = get_best_fit(name_chain, samples, base_path, #post=False,  irsic=False, kmax = False, noise=True)


# Plot bf for noise without bf parameters:
chi42_value, chi46_value, chi5_value, bf_pred_42, bf_pred_46, bf_pred_5 = get_best_fit(name_chain, samples, base_path, post=False,  irsic=False, kmax = False, noise=False, use_noise=False)
#Irsic+24:
#name_chain = 'chain_bo_WDM_pRELICS_T0G_moC'
name_chain = '/data/og313/lyalpha/code_wdm/chains_irsic/chain_bo_WDM_pRELICS_T0G_moC_nofqso'
#name_chain = '/data/og313/lyalpha/code_wdm/chains_irsic/chain_bo_WDM_pRELICS_patchy_u0T0_mru0_moC'


# Get plot:
# save_path = '/data/og313/lyalpha/code_wdm/year2/30_01/Rsu0_post_kmax20_3bins_vs_irsic.pdf'
save_path = f"{name.split('fwdm_')[1]}_vs_irsic_{name_chain.split('pRELICS_')[1]}.pdf"
print('Figure saved at', save_path)
# analysis = '$Patchy + {R_{s}}^{cdm}$ $T_{0}$ prior'
analysis = r'Patchy + $\tau_{\mathrm{CMB}}$ + ${R_{s}}^{cdm}$ ($u_{0}$)'
#analysis = 'Default'
#analysis = 'Noise NO An(z)'
#analysis = r'Re-Patchy + $T_{0}$ + $\tau$'
#analysis ='Default'
analysis = 'New noise'
analysis = 'Default'
save_path = 'bf_fiducial_test.pdf'

save_path = 'bf_fiducial_08.pdf'
#save_path = None
#analysis = '$k$ > 0.1 s/km by 20%'
plot_best_fit(bf_pred_42, bf_pred_46, bf_pred_5, chi42_value, chi46_value, chi5_value, save_path, analysis, kmax=False)
