
#%%
from funcs import get_samples_smooth, get_alpha_vs_fwdm
from getdist import plots
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Choose analysis to plot:
name = 'chain_fwdm_fiducial'
analysis = f"Fiducial"
#analysis = f"${{R_s}}^{{\mathrm{{cdm}}}}$ ($u_0$) 20%"
directory = f'/data/og313/lyalpha/code_wdm/archive_chains/{name}'
burning = 0.8
samples = get_samples_smooth(directory, burning)


# Add more analysis here:
name = 'chain_fwdm_repatchy_Rsu0_post_tau'
#name = 'chain_fwdm_fiducial_Rsu0_post_tau'
directory = f'/data/og313/lyalpha/code_wdm/archive_chains/{name}'
burning = 0.3
samples2 = get_samples_smooth(directory, burning)

# Add more analysis here:
#name = 'chain_fwdm_noise_not0g'
#name = 'chain_fwdm_u0t0_t0g_lowcut_Rsu0_post'
#directory = f'/data/og313/lyalpha/code_wdm/archive_chains/{name}'
#burning = 0.8
#samples3 = get_samples_smooth(directory, burning)

# Add more analysis here:
#name = 'chain_fwdm_u0t0_t0g_Rsu0_post'
#directory = f'/data/og313/lyalpha/code_wdm/archive_chains/{name}'
#burning = 0.8
#samples4 = get_samples_smooth(directory, burning)


samples_list = [samples, samples2]
#legend_list = ['Fiducial new']#, "",  '${R_{s}}^{cdm}$ ($u_{0}$) 20%']#, '${R_{s}}^{cdm}$ ($u_{0}$)']#'$k$ > 0.1 s/km by 10%']#
color_list = ['tab:blue', 'tab:orange']

#color_list = cm.Blues(np.linspace(0.3, 1, 2)).tolist()
#legend_list = [ r'$T_{0}$ prior', r'$T_{0}$ prior, $k_{\mathrm{max}} < 0.1$ s/km',
#                r'$u_{0}-T_{0}$ prior, $k_{\mathrm{max}} <$ 0.1 s/km']

legend_list = ['Default', r'Patchy + $\tau_{\mathrm{CMB}}$ + ${R_{s}}^{cdm}$ ($u_{0}$)']
#legend_list = ['Default',r' $\tau_{\mathrm{CMB}}$ + ${R_{s}}^{cdm}$ ($u_{0})$',
#               r'$u_{0}-T_{0}$ + ${R_{s}}^{cdm}$ ($u_{0}$']
#legend_list = [r'Patchy + $\tau_{\mathrm{CMB}}$ + ${R_{s}}^{cdm}$ ($u_{0}$)']
#legend_list = ['Fiducial']
      #          r'Patchy + $T_{0}$ + $\tau_{\mathrm{CMB}}$ + ${R_{s}}^{cdm}$ ($u_{0}$)']
          #     r'Patchy + $u_{0}-T_{0}$ + ${R_{s}}^{cdm}$ ($u_{0}$)'] # + ${R_{s}}^{\mathrm{cdm}}$ ($u_{0}$)']
save_path = '2d_alpha_fwdm_draft.pdf'
print('Saved in', save_path)
chain_irsic = '/data/og313/lyalpha/code_wdm/chains_irsic/chain_bo_WDM_pRELICS_T0G_moC_nofqso' 

#NOTE: to include more samples than the fiducial, and more legends. need to modify the function
get_alpha_vs_fwdm(samples_list, legend_list, color_list, save_path, chain_irsic)
# %%
