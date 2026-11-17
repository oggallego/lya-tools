#%%
from funcs import get_samples_smooth, get_samples_irsic, process_filtered_samples, get_1d_fwdm
from getdist import plots
import matplotlib.pyplot as plt

# Chose analysis and corresponding analysis from Vid
#chain_irsic = '/data/vault/og313/chains_irsic/chain_bo_WDM_pRELICS_patchy_T0G_mru0_moC'
#samples_irsic = get_samples_irsic(chain_irsic)


# Fiducial analysis:
name = 'chain_fwdm_fiducial'
#name = 'chain_fwdm_u0T0_T0G_nocut'
#name = 'chain_fwdm_t0g_tau'
#name = 'chain_fwdm_t0g_Rscdm_kmax'
#name = 'chain_fwdm_repatchy_Rsu0_post_tau'
#name = 'chain_fwdm_repatchy_u0t0_t0g_Rsu0_post_lowcut'
#name = 'chain_fwdm_fiducial_Rsu0_post_tau'
#name = 'chain_fwdm_u0T0_T0G_nocut'
#name = 'chain_fwdm_repatchy_u0t0_t0g'
#name = 'chain_fwdm_u0t0_t0g_Rsu0_post'
#name = 'chain_fwdm_noise_not0g_gaussian_new_ext'
#name = 'chain_fwdm_not0g_Rscdm_kmax'
#name = 'chain_fwdm_noise_t0g_ext_gaussian0.2'
#name = 'chain_fwdm_noise_t0g_ext_tau_gaussian0.2'
#name = 'chain_fwdm_noise_t0g_ext_u0t0_gaussian0.2'
#name = 'noise_runs/chain_fwdm_ext_big_t0g'
#name = 'noise_runs/chain_fwdm_noise_lognorm_clip2'
#name = 'chain_fwdm_u0t0_t0g_lowcut_Rsu0_post'
#name = 'chain_fwdm_fiducial_Rsu0_post_tau'
analysis = f"Noise"
directory = f'/data/og313/lyalpha/code_wdm/archive_chains/{name}'
burning = 0.8
samples = get_samples_smooth(directory, burning)
filtered_fwdm_1 = process_filtered_samples(samples, 'fwdm', 'alpha',  1, irsic=False, tolerance = 1e-5)

# Add more analysis here:
#name = 'chain_fwdm_repatchy_Rsu0_post_tau'
#directory = f'/data/og313/lyalpha/code_wdm/archive_chains/{name}'
#burning = 0.8
#samples2 = get_samples_smooth(directory, burning)
#filtered_fwdm_1_2 = process_filtered_samples(samples2, 'fwdm', 'alpha',  1, irsic=False)

# Add more analysis here:
# name = 'chain_fwdm_repatchy_Rsu0_post_tau'
# directory = f'/data/og313/lyalpha/code_wdm/archive_chains/{name}'
# burning = 0.8
# samples3 = get_samples_smooth(directory, burning)
# filtered_fwdm_1_3 = process_filtered_samples(samples3, 'fwdm', 'alpha',  1, irsic=False)

# Add more analysis here:
# name = 'chain_fwdm_patchy_u0t0_t0g_Rsu0_post_tau'
# directory = f'/data/og313/lyalpha/code_wdm/archive_chains/{name}'
# burning = 0.8
# samples4 = get_samples_smooth(directory, burning)
# filtered_fwdm_1_4 = process_filtered_samples(samples4, 'fwdm', 'alpha',  1, irsic=False)


color_list = ['tab:orange', 'tab:blue', 'tab:green', 'black']
#legend_list = [ r'$T_{0}$ prior', r'$T_{0}$ prior, $k_{\mathrm{max}} < 0.1$ s/km',
#                r'$u_{0}-T_{0}$ prior, $k_{\mathrm{max}} <$ 0.1 s/km']

legend_list = [ r'$T_{0}$', r'Patchy + $u_{0}$-$T_{0}$',
                r'Patchy + $u_{0}$-$T_{0}$ + ${R_{s}}^{cdm}$ ($u_{0}$)',
                r'$u_{0}-T_{0}$ + $\tau_{\mathrm{CMB}}$ + ${R_{s}}^{cdm}$ ($u_{0}$)', 'Irsic+24'] # + ${R_{s}}^{\mathrm{cdm}}$ ($u_{0}$)']
save_path = 'repatchy_Rsu0_post_1d_alpha.pdf'

legend_list= ['Fiducial', 'Noise']

legend_list = [ r'$T_{0}$',
                r'RE-Patchy + $T_{0}$ + ${R_{s}}^{cdm}$ ($u_{0}$)', 'Irsic+24']

# Get 1,2,3 sigma constraints:
sigma = 1-0.682689492137086

### more accurately
two_sigma = 1-0.954499736103642                                                                    #
three_sigma = 1-0.997300203936740
four_sigma = 1-0.999936657516334# = 4sigma
five_sigma = 1-0.999999426696856 #= 5sigma


import numpy as np

# Dont need to process Irsic+24 samples:

#save_path = 'test_pf.pdf'
g = plots.get_single_plotter(width_inch=8)#
fontsize = 20
g.settings.title_limit_fontsize = fontsize
g.settings.axes_fontsize=fontsize
g.settings.axes_labelsize=fontsize
g.settings.legend_fontsize=fontsize
g.settings.fontsize = 15
g.settings.legend_frame = False

# Plotting the samples with specified colors
#Note that the 2sigma constraints shown above correspond to filtered_fwdm_1 
g.plot_1d([filtered_fwdm_1], 'alpha', title_limit=2, colors=color_list)  # First dataset
#g.plot_1d(filtered_fwdm_1, 'alpha', title_limit=2, colors=['g'])  # Second dataset

# Adding the legend without the colors argument
g.add_legend(legend_list, legend_loc='upper left', bbox_to_anchor=(1.05, 1))

# Manually settinolors of the legend text
#for text, color in zip(legend.get_texts(), ['k', 'tab:orange']):
#            text.set_color(color)
if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

# from getdist import MCSamples
# %matplotlib inline
# samples = samples_irsic
# irsic = True
# fixed_param = 'fwdm'
# fixed_value = 1
# tolerance = 0.1

# # print((samples.samples))

# alpha_samples = samples.getParams().alpha
# if irsic ==False:
#     fwdm_samples = samples.getParams().fwdm
# weights = samples.weights
# loglikes = samples.loglikes

# if fixed_param == 'alpha':
#     #filtered_indices = np.abs(alpha_samples - fixed_value) < tolerance 
#     #print(f"Filtered indices sum: {np.sum(filtered_indices)}")
   
#     filtered_indices = np.abs(alpha_samples - fixed_value) < tolerance
#     filtered_fwdm_samples = fwdm_samples[filtered_indices]

#     n_samples = int(np.shape(filtered_fwdm_samples)[0])

#     # print('Number of filtered samples is', np.shape(filtered_fwdm_samples))
#     # print(f"Filtered indices sum: {np.sum(filtered_indices)}")
#     filtered_weights = weights[filtered_indices]
#     filtered_loglikes = loglikes[filtered_indices]

#     filtered_samples = MCSamples(
#     samples=np.array([filtered_fwdm_samples]).T,  # Transpose to make it 2D
#     names=['fwdm'],  labels = [r'f_{\mathrm{WDM}}'],# Parameter name
#     ranges = {'fwdm': [0,1]}, weights = filtered_weights, loglikes = filtered_loglikes)# LaTeX-style label for plotting

# if fixed_param == 'fwdm':
#     if irsic==False:
#         filtered_indices = np.abs(fwdm_samples - fixed_value) < tolerance 
#         filtered_alpha_samples = alpha_samples[filtered_indices]

#         print('Number of filtered samples is', np.shape(filtered_alpha_samples))
#         n_samples = int(np.shape(filtered_alpha_samples)[0])

#         filtered_weights = weights[filtered_indices]
#         filtered_loglikes = loglikes[filtered_indices]

#         filtered_samples = MCSamples(
#         samples=np.array([filtered_alpha_samples]).T,  # Transpose to make it 2D
#         names=['alpha'],  # Parameter name
#         labels=[r'{m_{\mathrm{WDM}}}^{-1}\,\mathrm{(keV^{-1})}'],  ranges = {'alpha': [0,1]}, weights = filtered_weights, loglikes = filtered_loglikes)# LaTeX-style label for plotting


#     if irsic==True:
#         # We do not have to filter for samples with f = 1 since all in Irsic+24 will satisfy this

#         filtered_samples = samples

# from getdist import MCSamples, plots
# import matplotlib.pyplot as plt
# # Extract parameter names and labels, excluding 'neff'
# param_names = [p.name for p in filtered_fwdm_1_other.paramNames.names if p.name != "neff"]
# param_labels = [p.label for p in filtered_fwdm_1_other.paramNames.names if p.name != "neff"]

# # Get indices of parameters excluding 'neff'
# param_indices = [i for i, p in enumerate(filtered_fwdm_1_other.paramNames.names) if p.name != "neff"]

# # Extract only relevant samples (without 'neff')
# filtered_samples_no_neff = filtered_fwdm_1_other.samples[:, param_indices]

# # Create a new MCSamples object
# filtered_samples_clean = MCSamples(
#     samples=filtered_samples_no_neff,
#     names=param_names,
#     labels=param_labels,
#     weights=filtered_fwdm_1_other.weights,
#     loglikes=filtered_fwdm_1_other.loglikes
# )
# alpha_min = 0
# alpha_max = 1
# # Now plot
# g = plots.get_single_plotter(width_inch=8)
# g.plot_1d(filtered_samples_clean, 'alpha', title_limit=2, ax=0, colors=['r', 'g', 'b'])
# g.plot_1d(filtered_fwdm_1, 'alpha', title_limit=2, ax=0, colors=['r', 'g', 'b'])
# plt.xlim(alpha_min, alpha_max)


# %%
