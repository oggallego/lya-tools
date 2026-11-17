#%%
from funcs import get_samples_smooth, get_u0_2d
from getdist import plots
import matplotlib.pyplot as plt

# Choose analysis to plot:
name = 'chain_fwdm_fiducial'
analysis = f"Fiducial"
#analysis = f"${{R_s}}^{{\mathrm{{cdm}}}}$ ($u_0$) 20%"
directory = f'/data/og313/lyalpha/code_wdm/archive_chains/{name}'
burning = 0.6
samples = get_samples_smooth(directory, burning)


# Add more analysis here:
name = 'chain_fwdm_patchy_Rsu0_post_tau'
directory = f'/data/og313/lyalpha/code_wdm/archive_chains/{name}'
burning = 0.3
samples2 = get_samples_smooth(directory, burning)

# Add more analysis here:


# Add more analysis here:

samples_list = [samples, samples2] 

# Add more analysis here:
#name = 'chain_fwdm_u0t0_t0g_Rscdm_kmax'
#directory = f'/data/og313/lyalpha/code_wdm/archive_chains/{name}'
#burning = 0.8
#samples3 = get_samples_smooth(directory, burning)


#legend_list = ['Fiducial new']#, "",  '${R_{s}}^{cdm}$ ($u_{0}$) 20%']#, '${R_{s}}^{cdm}$ ($u_{0}$)']#'$k$ > 0.1 s/km by 10%']#
color_list = ['tab:blue', 'tab:orange'] #'tab:green']
#legend_list = [ r'$T_{0}$ prior', r'$T_{0}$ prior, $k_{\mathrm{max}} < 0.1$ s/km',
#                r'$u_{0}-T_{0}$ prior, $k_{\mathrm{max}} <$ 0.1 s/km']
legend_list = [ r'Default', 
                r'Patchy + $\tau_{\mathrm{CMB}}$ + ${R_{s}}^{cdm}$ ($u_{0}$)']
#legend_list = [ r'$T_{0}$', r'Patchy + $T_{0}$ + ${R_{s}}^{cdm}$ ($u_{0}$)',
# r'RE-Patchy + $T_{0}$ + ${R_{s}}^{cdm}$ ($u_{0}$)']
      #          r'Patchy + $T_{0}$ + $\tau_{\mathrm{CMB}}$ + ${R_{s}}^{cdm}$ ($u_{0}$)']
          #     r'Patchy + $u_{0}-T_{0}$ + ${R_{s}}^{cdm}$ ($u_{0}$)'] # + ${R_{s}}^{\mathrm{cdm}}$ ($u_{0}$)']
save_path = '/data/og313/lyalpha/code_wdm/paper/draft/2d_u0_draft.pdf'
print('Saved in', save_path)


print('Saved in', save_path)

#chain_irsic = 'chain_bo_WDM_pRELICS_T0G_mru0_moC'
chain_irsic = '/data/og313/lyalpha/code_wdm/chains_irsic/chain_bo_WDM_pRELICS_u0T0_moC'

get_u0_2d(samples_list, color_list, legend_list, save_path, chain_irsic)
# %%
