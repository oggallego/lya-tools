#%%
from funcs import get_samples_smooth, get_samples_irsic, process_filtered_samples, get_1d_fwdm
from getdist import plots
import matplotlib.pyplot as plt
import numpy as np

'No Irsic+24 loaded since can not compare for fwdm constraints'


# Fiducial analysis:
name = 'chain_fwdm_fiducial'
#name = 'chain_fwdm_u0T0_T0G_nocut'
#name = 'chain_fwdm_t0g_tau'
#name = 'chain_fwdm_repatchy_Rsu0_post_tau'
name = 'chain_fwdm_t0g_Rscdm_kmax'
#name = 'chain_fwdm_not0g_Rscdm_kmax'
#name = 'noise_runs/chain_fwdm_noise_lognorm_clip2'

name = 'chain_fwdm_fiducial_Rsu0_post_tau'
#name = 'chain_fwdm_patchy_Rscdm'
#name = 'chain_fwdm_repatchy_Rsu0_post_tau'
#name = 'chain_fwdm_repatchy_u0t0_t0g_Rsu0_post_lowcut'
#name = 'chain_fwdm_fiducial_Rsu0_post_tau'
#name = 'chain_fwdm_repatchy_u0t0_t0g'

name = 'chain_fwdm_u0t0_t0g_Rsu0_post'
#name = 'chain_fwdm_noise_not0g_gaussian_postRs'
#name = 'noise_runs/chain_fwdm_noise_t0g_gaussian0.2'
#name = 'noise_runs/chain_fwdm_noise_gaussian0.2_bigstep'
#name = 'chain_fwdm_not0g_Rscdm_kmax'
#name = 'noise_runs/chain_fwdm_ext_big_t0g'


#name = 'chain_fwdm_u0t0_t0g_lowcut_Rsu0_post'

#name = 'chain_fwdm_repatchy_u0t0_t0g'
analysis = f"Fiducial"
#analysis = f"${{R_s}}^{{\mathrm{{cdm}}}}$ ($u_0$) 20%"
directory = f'/data/og313/lyalpha/code_wdm/archive_chains/{name}'
burning = 0.9
samples = get_samples_smooth(directory, burning)
#tolerance = 1e-5
filtered_alpha_1 = process_filtered_samples(samples, 'alpha', 'fwdm',  1, irsic=False, tolerance=1e-4)
filtered_alpha_05 = process_filtered_samples(samples, 'alpha', 'fwdm',  1/2, irsic=False, tolerance=1e-4)
filtered_alpha_03 = process_filtered_samples(samples, 'alpha', 'fwdm',  1/3, irsic=False, tolerance=1e-4)
filtered_alpha_025 = process_filtered_samples(samples, 'alpha', 'fwdm',  1/4, irsic=False, tolerance=1e-4)
filtered_alpha_02 = process_filtered_samples(samples, 'alpha', 'fwdm',  1/5, irsic=False, tolerance=1e-4)

# What about pure WDM constraint?
filtered_fwdm_1 = process_filtered_samples(samples, 'fwdm', 'alpha',  1, irsic=False, tolerance = 1e-4)
# Add more analysis here:
#name = 'chain_fwdm_not0g_Rscdm_kmax'
#name = 'chain_fwdm_u0t0_t0g_Rsu0_post'
#name = 'chain_fwdm_repatchy_Rsu0_post_tau'
#name = 'noise_runs/chain_fwdm_noise_lognorm'

#directory = f'/data/og313/lyalpha/code_wdm/archive_chains/{name}'
#burning = 0.8
#samples2 = get_samples_smooth(directory, burning)
#filtered_alpha_1_2 = process_filtered_samples(samples2, 'alpha', 'fwdm',  1, irsic=False)
#filtered_alpha_05_2 = process_filtered_samples(samples2, 'alpha', 'fwdm',  0.5, irsic=False)
#filtered_alpha_03_2 = process_filtered_samples(samples2, 'alpha', 'fwdm',  1/3, irsic=False)
#filtered_alpha_025_2 = process_filtered_samples(samples2, 'alpha', 'fwdm',  1/4, irsic=False)

# Add more analysis here:
#name = 'chain_fwdm_t0g_noise'
#directory = f'/data/og313/lyalpha/code_wdm/archive_chains/{name}'
#burning = 0.8
#samples3 = get_samples_smooth(directory, burning)
#filtered_alpha_1_3 = process_filtered_samples(samples3, 'alpha', 'fwdm',  1, irsic=False)
#filtered_alpha_05_3 = process_filtered_samples(samples3, 'alpha', 'fwdm',  0.5, irsic=False)
#filtered_alpha_03_3 = process_filtered_samples(samples3, 'alpha', 'fwdm',  1/3, irsic=False)
#filtered_alpha_025 = process_filtered_samples(samples, 'alpha', 'fwdm',  1/4, irsic=False)

#name = 'chain_fwdm_u0t0_t0g_Rsu0_post'
#directory = f'/data/og313/lyalpha/code_wdm/archive_chains/{name}'
#burning = 0.8
#samples4 = get_samples_smooth(directory, burning)
#filtered_alpha_1_4 = process_filtered_samples(samples4, 'alpha', 'fwdm',  1, irsic=False)
#filtered_alpha_05_4 = process_filtered_samples(samples4, 'alpha', 'fwdm',  0.5, irsic=False)
#filtered_alpha_03_4 = process_filtered_samples(samples4, 'alpha', 'fwdm',  1/3, irsic=False)


legend_list = [ r'$T_{0}$ prior', r'$T_{0}$ prior, $k_{\mathrm{max}} < 0.1$ s/km',
                r'$u_{0}-T_{0}$ prior, $k_{\mathrm{max}} <$ 0.1 s/km']
legend_list = [ r'$T_{0}$ prior',r'Patchy + $T_{0}$ prior',  r'Patchy + $u_{0}-T_{0}$ prior'] # + ${R_

legend_list = [ r'$T_{0}$', r'$T_{0}$ + ${R_{s}}^{cdm}$ ($u_{0}$)',
                r'$T_{0}$ + $\tau_{\mathrm{CMB}}$ + ${R_{s}}^{cdm}$ ($u_{0}$)',
                r'$u_{0}-T_{0}$ + ${R_{s}}^{cdm}$ ($u_{0}$)'] # + ${R_{s}}^{\mathrm{cdm}}$ ($u_{0}$)']
save_path = 'Rsu0_post_1d_fwdm.pdf'
save_path = 'noise_1d_fwdm_test.pdf'
save_path = None
legend_list= ['Fiducial', 'Repatchy']


color_list = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red']

# Get 1,2,3 sigma constraints:
sigma = 1-0.682689492137086

### more accurately
two_sigma = 1-0.954499736103642                                                                    #
three_sigma = 1-0.997300203936740
four_sigma = 1-0.999936657516334# = 4sigma
five_sigma = 1-0.999999426696856 #= 5sigma



g = plots.get_single_plotter(width_inch=16)#
fontsize = 20
g.settings.title_limit_fontsize = fontsize
g.settings.axes_fontsize=fontsize
g.settings.axes_labelsize=fontsize
g.settings.legend_fontsize=fontsize
g.settings.fontsize = fontsize
g.settings.legend_frame = False

g.make_figure(nplot=3, nx=3, ny=1, xstretch=9, ystretch=10)  # Increase xstretch for more horizontal space
plt.subplots_adjust(wspace=1.0, hspace=0.8)  # Increase wspace for horizontal separation

g.plot_1d([filtered_alpha_1, filtered_alpha_1_2], 'fwdm', ax=0, title_limit=2, colors=color_list)
g.add_text(r'${m_{\mathrm{WDM}}}$ = 1 keV', 0.97, 0.9)
#g.add_legend(legend_list)

g.plot_1d([filtered_alpha_05, filtered_alpha_05_2, filtered_alpha_05_3], 'fwdm', ax=1, title_limit=2, colors=color_list)
g.add_text(r'${m_{\mathrm{WDM}}}$ = 2 keV', 0.97, 0.9)


g.plot_1d([filtered_alpha_03, filtered_alpha_03_2, filtered_alpha_03_3], 'fwdm', ax=2, title_limit=2, colors=color_list)
g.add_text(r'${m_{\mathrm{WDM}}}$ = 3 keV', 0.97, 0.9)

g.add_legend(legend_list, legend_loc='upper left', bbox_to_anchor=(1.05, 1))

if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
