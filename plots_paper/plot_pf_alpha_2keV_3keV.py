#%%
from funcs import get_samples_smooth, get_samples_irsic, process_filtered_samples, get_1d_fwdm
from getdist import plots
import matplotlib.pyplot as plt
import numpy as np

'No Irsic+24 loaded since can not compare for fwdm constraints'


# Fiducial analysis:
name = 'chain_fwdm_fiducial'
analysis = f"Fiducial"
#analysis = f"${{R_s}}^{{\mathrm{{cdm}}}}$ ($u_0$) 20%"
directory = f'/data/og313/lyalpha/code_wdm/archive_chains/{name}'
burning = 0.8
tolerance = 1e-3
samples = get_samples_smooth(directory, burning)
# filtered_alpha_1 = process_filtered_samples(samples, 'alpha', 'fwdm',  1, irsic=False)
filtered_alpha_05 = process_filtered_samples(samples, 'alpha', 'fwdm',  1/2, irsic=False, tolerance=1e-6)
filtered_alpha_03 = process_filtered_samples(samples, 'alpha', 'fwdm',  1/3, irsic=False, tolerance=1e-5)
# filtered_alpha_025 = process_filtered_samples(samples, 'alpha', 'fwdm',  1/4, irsic=False)
# filtered_alpha_02 = process_filtered_samples(samples, 'alpha', 'fwdm',  1/5, irsic=False)



# %%
