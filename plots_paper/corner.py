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
burning = 0.9
samples = get_samples_smooth(directory, burning)

# Add more analysis here:
name = 'chain_fwdm_repatchy_Rsu0_post_tau'
#name = 'chain_fwdm_fiducial_Rsu0_post_tau'
directory = f'/data/og313/lyalpha/code_wdm/archive_chains/{name}'
burning = 0.8
samples2 = get_samples_smooth(directory, burning)

color_list = ['tab:blue', 'tab:orange']
legend_list = ['Fiducial', r'Patchy + $\tau_{\mathrm{CMB}}$ + ${R_{s}}^{cdm}$ ($u_{0}$)']

g = plots.get_subplot_plotter(subplot_size=50, width_inch=60)
print('Changes saved')
g.settings.axes_fontsize=37
g.settings.axes_labelsize=37
g.settings.legend_fontsize=45
g.settings.fontsize = 37
g.settings.legend_frame = False
g.settings.alpha_factor_contour_lines=0.45
g.settings.alpha_filled_add = 0.55

#g = plots.get_subplot_plotter(subplot_size=20, width_inch=30)
g.triangle_plot([samples, samples2], samples.getParamNames().list()[:-4], colors = ['tab:blue', 'tab:orange'], filled = True, line_args=[{'color':'tab:blue'}, {'color':'tab:orange'}], legend_labels = legend_list, legend_loc='upper right')
#g.add_legend(legend_list,  legend_loc='upper right')
g.export('/data/og313/lyalpha/code_wdm/paper/draft/corner_appendix.pdf')
