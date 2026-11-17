#%%
import numpy as np
import plotting_library as PL
from pylab import *
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

# Snapshots
snapshot_cdm = '/data/emergence13/DiRAC_relics/planck1_20_1024/snapdir_009/snap_009'
snapshot_wdm2 = '/data/emergence9/og313/DiRAC_relics_CWDM/planck1_20_1024_wdm1/snapdir_009/snap_009'
snapshot_wdm4 = '/data/emergence10/prace_relics/planck1_20_1024_wdm3/snapdir_009/snap_009'

# Density field parameters
x_min, x_max = 0, 20
y_min, y_max = 0, 20
z_min, z_max = 10, 11
grid = 1024
plane = 'XY'
MAS = 'TSC'
save_df = False

# Image parameters
min_overdensity = 0.1
max_overdensity = 20
scale = 'log'
cmap = 'viridis'

# Titles for the subplots
titles = ['CDM gas', 'CDM DM', 'WDM 1 keV DM', 'WDM 3 keV DM']

# Colormaps for each plot
colormaps = ['viridis', 'inferno', 'inferno', 'inferno']  # Same colormap for WDM2 and WDM4


# Initialize the figure
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Compute and plot the density fields
snapshots = [(snapshot_cdm, [0]),  # CDM gas
             (snapshot_cdm, [1]),  # CDM DM
             (snapshot_wdm2, [1]), # WDM 1 keV DM
             (snapshot_wdm4, [1])] # WDM 4 keV DM

# Store color axis handles for colorbars
cax_row1 = None
cax_row2 = []

cax_row1 = None
cax_row2 = []

for i, (ax, (snapshot, ptypes), cmap, title) in enumerate(zip(axs.flatten(), snapshots, colormaps, titles)):
    dx, x, dy, y, overdensity = PL.density_field_2D(snapshot, x_min, x_max, y_min, y_max,
                                                    z_min, z_max, grid, ptypes, plane, MAS, save_df)
    overdensity[overdensity < min_overdensity] = min_overdensity

    if scale == 'linear':
        cax = ax.imshow(overdensity, cmap=cmap, origin='lower',
                        extent=[x, x + dx, y, y + dy], interpolation='bicubic',
                        vmin=min_overdensity, vmax=max_overdensity)
    else:
        cax = ax.imshow(overdensity, cmap=cmap, origin='lower',
                        extent=[x, x + dx, y, y + dy], interpolation='bicubic',
                        norm=LogNorm(vmin=min_overdensity, vmax=max_overdensity))

    if i < 2:
        if cax_row1 is None:
            cax_row1 = cax
    else:
        cax_row2.append(cax)

    ax.text(1, 18, title, color='white', fontsize=15, weight='bold', ha='left', va='top',
            bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))
    ax.set_xticks([])
    ax.set_yticks([])

# Adjust layout to leave space on the left
fig.subplots_adjust(left=0.15)

# Colorbar for first row (left side, aligned to row 1)
cbar_ax1 = fig.add_axes([0.1, 0.54, 0.03, 0.44])  # [left, bottom, width, height]
cbar1 = fig.colorbar(cax_row1, cax=cbar_ax1, orientation='vertical')
# cbar1.set_label(r"$\rho/\bar{\rho}$ Gas", fontsize=14, labelpad=10)
cbar1.set_label(r"1 + $\delta_{\mathrm{gas}}$", fontsize=16, labelpad=12)
cbar1.ax.yaxis.set_label_position('left')  # Move label to the right
cbar1.ax.tick_params(labelsize=14)

# Colorbar for second row (left side, aligned to row 2)
cbar_ax2 = fig.add_axes([0.1, 0.04 , 0.03, 0.45])
cbar2 = fig.colorbar(cax_row2[0], cax=cbar_ax2, orientation='vertical')
cbar2.set_label(r"1 + $\delta_{\mathrm{DM}}$", fontsize=16, labelpad=12)
cbar2.ax.yaxis.set_label_position('left')  # Move label to the right
cbar2.ax.tick_params(labelsize=14)

# Final layout and save
plt.tight_layout(rect=[0.15, 0, 1, 1])  # Leave space on the left for colorbars
plt.savefig("/data/og313/lyalpha/code_wdm/paper/draft/combined_density_field_z42.pdf", bbox_inches='tight')
plt.show()
# %%
