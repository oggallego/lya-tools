

name = 'chain_fwdm_fiducial'
name = 'chain_fwdm_u0T0_T0G_nocut'
name = 'chain_fwdm_fiducial_Rsu0_post_tau'

name = 'chain_fwdm_u0t0_t0g_lowcut_Rsu0_post'
#name = 'chain_fwdm_t0g_tau'
#name = 'chain_fwdm_t0g_Rscdm_kmax'
#name = 'chain_fwdm_noise_not0g'
#name = 'chain_fwdm_patchy_Rscdm'
#name = 'chain_fwdm_repatchy_Rsu0_post_tau'
#name = 'chain_fwdm_repatchy_u0t0_t0g_Rsu0_post_lowcut'
#name = 'chain_fwdm_fiducial_Rsu0_post_tau'
#name = 'chain_fwdm_repatchy_u0t0_t0g'
#name = 'chain_fwdm_u0t0_t0g_Rsu0_post'
#name = 'chain_fwdm_noise_not0g_gaussian_postRs'
#name = 'noise_runs/chain_fwdm_noise_t0g_gaussian0.2'
#name = 'noise_runs/chain_fwdm_noise_gaussian0.2_bigstep'
#name = 'chain_fwdm_not0g_Rscdm_kmax'
#name = 'noise_runs/chain_fwdm_ext_big_t0g'
#name = 'noise_runs/chain_fwdm_noise_lognorm_clip2'
#name = 'chain_fwdm_not0g_Rscdm_kmax

print(name)
from getdist import loadMCSamples
def get_samples_smooth(directory, burning):
    """
    Get samples from array job - chain name format format from _{i}.1.txt and dirrctory
    Directory needs to contain chain.paramnames and chain.ranges
    """
    smooth_2D = 0.6
    smooth_1D = 0.6
    ignore_rows = burning 

    #
    # Load the samples without altering parameter names yet
    # To run this need .chainparamnames in the same folder
    samples_smooth = loadMCSamples(directory, settings={
        'ignore_rows': ignore_rows,
        'smooth_scale_2D': smooth_2D,
        'smooth_scale_1D': smooth_1D
    })
    print('Gelman-Rubin ratio', samples_smooth.getGelmanRubin())
    total_samples = samples_smooth.samples.shape[0] #ids 1 corresponds to number of parameters
    print('Total number of samples is', total_samples)

    return samples_smooth

directory = f'/data/og313/lyalpha/code_wdm/archive_chains/{name}'
burning = 0.2
samples = get_samples_smooth(directory, burning)

# Mean and best-fit of the posteriors:
stats = samples.getLikeStats()
# stats = samples.getMargeStats()

n_pars = 14
sigma = 1-0.682689492137086
arr = [stats.names[i].bestfit_sample for i in range(n_pars)]
# arr = [stats.names[i].mean for i in range(n_pars)]
# print('Best-fit parameters for An are', arr[0], arr[1], arr[-3])
t0z42 = arr[0]
t0z46 = arr[1]
t0z5 = arr[2]

onesigma = []
t0z_params = ['T0z46']

# Create a dictionary to map parameter names to their indices
param_index = {stats.names[i].name: i for i in range(n_pars)}
print('Best-fit parameters for T0 are', arr[0], arr[1], arr[2])

param_index = {stats.names[i].name: i for i in range(n_pars)}
print('Best-fit parameters for T0 are', arr[0], arr[1], arr[2])
# for param in t0z_params:
upper_sigma = (samples.confidence('T0z46', limfrac=sigma/2, upper=True))
lower_sigma = (samples.confidence('T0z46', limfrac=sigma/2, upper=False))
print('Lower and upper limit to add to table',stats.names[1].bestfit_sample - lower_sigma, upper_sigma - stats.names[1].bestfit_sample)

#print(samples.getLikeStats())
low = samples.paramNames.parWithName(f'T0z46').ND_limit_bot[0]
print('Low', low)
up = samples.paramNames.parWithName('T0z46').ND_limit_top[0]
#print(stats.names[1].bestfit_sample)
print(stats.names[1].bestfit_sample - low, up - stats.names[1].bestfit_sample)


if 'lowcut' in name:
    upper_sigma = (samples.confidence('T0z46', limfrac=sigma, upper=True))
    #lower_sigma = (samples.confidence('u0z46', limfrac=sigma, upper=False))
    print('Skewed upper sigma', upper_sigma)
#    lims = stats.parWithName(param).limits
#    for conf, lim in zip(samples.contours, lims):
#        if conf == 0.68:
#            print((lim.lower, lim.upper))
#            idx = param_index[param]
#            print((stats.names[idx].bestfit_sample - lim.lower, lim.upper - stats.names[idx].bestfit_sample))  # Append as a tuple

u0z_params = ['u0z46']
#print('Best-fit parameters for u0 are', arr[9], arr[10], arr[11])
if 'lowcut' in name or 'Rsu0_post_tau' in name or '_repatchy_u0t0_t0g_Rsu0_post' in name:
    upper_sigma = (samples.confidence('u0z46', limfrac=sigma/2, upper=True))
    lower_sigma = (samples.confidence('u0z46', limfrac=sigma, upper=False))
    print('Lower and upper limit to add to table', lower_sigma)
else:
 upper_sigma = (samples.confidence('u0z46', limfrac=sigma/2, upper=True))
 lower_sigma = (samples.confidence('u0z46', limfrac=sigma/2, upper=False))
 print('Lower and upper limit to add to table',stats.names[10].bestfit_sample - lower_sigma, upper_sigma - stats.names[10].bestfit_sample)

#low = samples.paramNames.parWithName(f'u0z46').ND_limit_bot[0]
#up = samples.paramNames.parWithName('u0z46').ND_limit_top[0]
#print(stats.names[10].bestfit_sample - low, up - stats.names[10].bestfit_sample)

#for param in u0z_params:
#    lims = stats.parWithName(param).limits
#    for conf, lim in zip(samples.contours, lims):
#        if conf == 0.68:
#            print('Limits', (lim.lower, lim.upper))
#            idx = param_index[param]
#            print('Limits to add to the table', (stats.names[idx].bestfit_sample - lim.lower, lim.upper - stats.names[idx].bestfit_sample))  # Append as a tuple
