#%%
from funcs import get_samples_smooth, get_samples_irsic, get_2sigma_filtered_samples, get_1d_fwdm
from getdist import plots
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Chose analysis and corresponding analysis from Vid
# chain_irsic = '/data/og313/lyalpha/code_wdm/chains_irsic/chain_bo_WDM_pRELICS_patchy_T0G_mru0_moC'
# samples_irsic = get_samples_irsic(chain_irsic)


# Fiducial analysis:
name = 'chain_fwdm_fiducial'
analysis = f"Fiducial"
directory = f'/data/og313/lyalpha/code_wdm/archive_chains/{name}'
burning = 0.8
samples = get_samples_smooth(directory, burning)
tolerance = 1e-4
bound_alpha_1 = get_2sigma_filtered_samples(samples, 'alpha', 'fwdm', 1, irsic=False, tolerance=tolerance)
bound_alpha_05 = get_2sigma_filtered_samples(samples, 'alpha', 'fwdm', 1/2, irsic=False, tolerance=1e-3)
bound_alpha_03 = get_2sigma_filtered_samples(samples, 'alpha', 'fwdm', 1/3, irsic=False, tolerance=1e-3)
bound_alpha_025 = get_2sigma_filtered_samples(samples, 'alpha', 'fwdm', 1/4, irsic=False, tolerance=tolerance)


# Add more analysis here:
name = 'chain_fwdm_repatchy_u0t0_t0g_Rsu0_post_lowcut'
name = 'chain_fwdm_repatchy_Rsu0_post_tau'
print('Constraints from patchy + Rsu0 + tau')
directory = f'/data/og313/lyalpha/code_wdm/archive_chains/{name}'
burning = 0.3
samples2 = get_samples_smooth(directory, burning)
bound_alpha_1_2 = get_2sigma_filtered_samples(samples2, 'alpha', 'fwdm', 1, irsic=False, tolerance=0.1)
bound_alpha_05_2 = get_2sigma_filtered_samples(samples2, 'alpha', 'fwdm', 1/2, irsic=False, tolerance=0.15)
bound_alpha_03_2 = get_2sigma_filtered_samples(samples2, 'alpha', 'fwdm', 1/3, irsic=False, tolerance=1e-3)
bound_alpha_025_2 = get_2sigma_filtered_samples(samples2, 'alpha', 'fwdm', 1/4, irsic=False, tolerance=1e-3)


# Add more analysis here:
#name = 'chain_fwdm_t0g_Rscdm_kmax'
name = 'chain_fwdm_not0g_Rscdm_kmax'
directory = f'/data/og313/lyalpha/code_wdm/archive_chains/{name}'
burning = 0.8
samples3 = get_samples_smooth(directory, burning)
bound_alpha_1_3 = get_2sigma_filtered_samples(samples3, 'alpha', 'fwdm', 1, irsic=False, tolerance=tolerance)
bound_alpha_05_3 = get_2sigma_filtered_samples(samples3, 'alpha', 'fwdm', 1/2, irsic=False, tolerance=tolerance)
bound_alpha_03_3 = get_2sigma_filtered_samples(samples3, 'alpha', 'fwdm', 1/3, irsic=False, tolerance=tolerance)
bound_alpha_025_3 = get_2sigma_filtered_samples(samples3, 'alpha', 'fwdm', 1/4, irsic=False, tolerance=tolerance)

# Add more analysis here:
name = 'chain_fwdm_t0g_noise'
name = 'chain_fwdm_noise_t0g_ext_u0t0_gaussian0.2'
name = 'chain_fwdm_ext_big_t0g'
name = 'chain_fwdm_noise_lognorm_clip2'
directory = f'/data/og313/lyalpha/code_wdm/archive_chains/noise_runs/{name}'
burning = 0.8
samples4 = get_samples_smooth(directory, burning)
bound_alpha_1_4 = get_2sigma_filtered_samples(samples4, 'alpha', 'fwdm', 1, irsic=False, tolerance=tolerance)
bound_alpha_05_4 = get_2sigma_filtered_samples(samples4, 'alpha', 'fwdm', 1/2, irsic=False, tolerance=tolerance)
bound_alpha_03_4 = get_2sigma_filtered_samples(samples4, 'alpha', 'fwdm', 1/3, irsic=False, tolerance=tolerance)
bound_alpha_025_4 = get_2sigma_filtered_samples(samples4, 'alpha', 'fwdm', 1/4, irsic=False, tolerance=tolerance)

name = 'chain_fwdm_u0t0_t0g_lowcut_Rsu0_post'
directory = f'/data/og313/lyalpha/code_wdm/archive_chains/{name}'
burning = 0.8
print('For samples with u0t0 and Rsu0')
samples5 = get_samples_smooth(directory, burning)
bound_alpha_1_5 = get_2sigma_filtered_samples(samples5, 'alpha', 'fwdm',  1, irsic=False, tolerance=tolerance)
bound_alpha_05_5 = get_2sigma_filtered_samples(samples5, 'alpha', 'fwdm',  1/2, irsic=False , tolerance=tolerance)
bound_alpha_03_5 = get_2sigma_filtered_samples(samples5, 'alpha', 'fwdm',  1/3, irsic=False, tolerance=tolerance)
bound_alpha_025_5 = get_2sigma_filtered_samples(samples5, 'alpha', 'fwdm',  1/4, irsic=False, tolerance=1e-2)




analysis = [r'$k_{\mathrm{max}} <$ 0.1 s/km',
            r'$A_{\mathrm{noise}}$',
            r'Patchy + $\tau_{\mathrm{CMB}}$ + ${R_{s}}^{\mathrm{cdm}}$ ($u_{0}$)',
            r'$u_{0}-T_{0}$ + ${R_{s}}^{\mathrm{cdm}}$ ($u_{0}$)',
            'Fiducial' ]

# colors = cm.Blues(np.linspace(0.3, 1,5))

colors = cm.Blues(np.linspace(0.3, 0.8, len(analysis) - 1))

# Ensure colors is a 2D array before stacking
colors = np.array(colors)
print(np.shape(colors))

# Add an orange color for the last analysis
colors = np.vstack([colors, [1, 0.5, 0, 1]])
print(np.shape(colors))

print(len(colors))  # Should print 5

fontsize = 25
plt.rcParams["font.size"] = fontsize   # Global font size
plt.rcParams["axes.labelsize"] = fontsize  # Axis labels
plt.rcParams["xtick.labelsize"] = fontsize   # X-axis tick labels
plt.rcParams["ytick.labelsize"] = fontsize   # Y-axis tick labels
plt.rcParams["legend.fontsize"] = 24   # Legend text

mwdm = [1,2,3,4]

print('Fiducial:', bound_alpha_025)
fwdm_bounds = {
    r'$k_{\mathrm{max}} <$ 0.1 s/km': (bound_alpha_1_3, bound_alpha_05_3, bound_alpha_03_3, bound_alpha_025_3),
    r'$A_{\mathrm{noise}}$':( bound_alpha_1_4, bound_alpha_05_4, bound_alpha_03_4, bound_alpha_025_4),
    r'$u_{0}-T_{0}$ + ${R_{s}}^{\mathrm{cdm}}$ ($u_{0}$)': ( bound_alpha_1_5, bound_alpha_05_5, bound_alpha_03_5, bound_alpha_025_5),
    'Fiducial': (bound_alpha_1, bound_alpha_05, bound_alpha_03, bound_alpha_025),
    r'Patchy + $\tau_{\mathrm{CMB}}$ + ${R_{s}}^{\mathrm{cdm}}$ ($u_{0}$)': ( bound_alpha_1_2, bound_alpha_05_2, bound_alpha_03_2, bound_alpha_025_2)
    # r'Patchy + $\tau_{\mathrm{CMB}}$ + ${R_{s}}^{cdm}$ ($u_{0}$)': ( bound_alpha_1_2, bound_alpha_05_2, bound_alpha_03_2, bound_alpha_025_2),
   
}
x = np.arange(len(mwdm))  # the label locations
width = 0.15  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(figsize=(14, 10))

for color, (analysis, bound) in zip(colors, fwdm_bounds.items()):
    offset = width * multiplier
    rects = ax.bar(x + offset, bound, width, label=analysis, color=color)
    # ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel(r'$f_{\mathrm{WDM}}$')
ax.set_xlabel(r'$m_{\mathrm{WDM}}$ (keV)')
ax.set_xticks(x + width * (len(fwdm_bounds) - 1) / 2)
ax.set_xticklabels(mwdm)
ax.set_ylim(0,1)
ax.legend(loc='upper left', frameon = False )
plt.savefig('/data/og313/lyalpha/code_wdm/paper/draft/2sigm_fwdm_draft.pdf')
plt.show()


# %%
