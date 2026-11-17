
#%%
from funcs import get_samples_smooth
import numpy as np
from getdist import plots
#from cobaya import load_samples
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.cm as cm



#Data taken from:
#https://arxiv.org/abs/2108.03699
#https://iopscience.iop.org/article/10.3847/1538-4357/aad4fd/pdf
#https://ui.adsabs.harvard.edu/abs/2013MNRAS.430.2067B/abstract



from funcs import get_samples_irsic, get_bf_sigma_irsic_all, get_sigma_irsic_alpha
name_chain= '/data/vault/og313/chains_irsic/chain_bo_WDM_pRELICS_patchy_T0G_mru0_moC'
samples_irsic = get_samples_irsic(name_chain)
bestfit, onesigma_all = get_bf_sigma_irsic_all(name_chain, samples_irsic)
one_sigma, two_sigma =  get_sigma_irsic_alpha(samples_irsic)

def get_taueff_bf(name):
    directory = f'/data/og313/lyalpha/code_wdm/archive_chains/{name}'
    burning = 0.8
    samples = get_samples_smooth(directory, burning)
    like_stats = samples.getLikeStats()
    stats = samples.getMargeStats()

    if 'noise' in name:
        n_pars = 17
    else:
        n_pars = 14
    
    # arr = [like_stats.names[i].bestfit_sample for i in range(n_pars)]
    arr = [stats.names[i].mean for i in range(n_pars)]
    print('Best-fit parameters for u0 are', arr[6], arr[7], arr[8])
    tauz42 = arr[6]
    tauz46 = arr[7]
    tauz5= arr[8]
    
    onesigma = []
    t0z_params = ['taueffz42', 'taueffz46', 'taueffz5']
    for i in t0z_params:
        lims = stats.parWithName(f'{i}').limits
        for conf, lims in zip(samples.contours,lims):
            if conf == 0.68:
                print((lims.lower, lims.upper))
                onesigma.append((lims.lower, lims.upper))  # Append as a tuple

    # Get 1 sigma error bars:
    # Get 1 sigma error bars:
    lower_error_t0z42 = max(tauz42 - onesigma[0][0], 0)
    upper_error_t0z42 = max(onesigma[0][1] - tauz42, 0)

    lower_error_t0z46 = max(tauz46 - onesigma[1][0], 0)
    upper_error_t0z46 = max(onesigma[1][1] - tauz46, 0)

    lower_error_t0z5 = max(tauz5 - onesigma[2][0], 0)
    upper_error_t0z5 = max(onesigma[2][1] - tauz5, 0)

    # Dict to get results:
    t0_dict = {
        't0z4.2': {
            'value': tauz42,
            'lower_error': lower_error_t0z42,
            'upper_error': upper_error_t0z42
        },
        't0z4.6': {
            'value': tauz46,
            'lower_error': lower_error_t0z46,
            'upper_error': upper_error_t0z46
        },
        't0z5.0': {
            'value': tauz5,
            'lower_error': lower_error_t0z5,
            'upper_error': upper_error_t0z5
        }
    }
    # Print information for each entry
    for key, value in t0_dict.items():
        print(f"For {key}:")
        print(f"  Best fit value: {value['value']}")
        print(f"  Lower error: {value['lower_error']}")
        print(f"  Upper error: {value['upper_error']}")
        print()
    
    return t0_dict


markersize = 3
fontsize = 20
plt.rcParams["font.size"] = fontsize   # Global font size
plt.rcParams["axes.labelsize"] = fontsize  # Axis labels
plt.rcParams["xtick.labelsize"] = fontsize   # X-axis tick labels
plt.rcParams["ytick.labelsize"] = fontsize   # Y-axis tick labels
plt.rcParams["legend.fontsize"] = 20   # Legend text
plt.figure(figsize = (10, 7))

x_values = [4.2, 4.6, 5.0]
diff = 0.03
t0_dicts = [
    get_taueff_bf('chain_fwdm_not0g_Rscdm_kmax'),
    # get_taueff_bf('/noise_runs/chain_fwdm_noise_lognorm'),
    # get_taueff_bf('chain_fwdm_u0t0_t0g_lowcut_Rsu0_post'),
    get_taueff_bf('chain_fwdm_fiducial'),
    get_taueff_bf('chain_fwdm_repatchy_Rsu0_post_tau')
    # Add more analyses as needed
]

analysis = [r'$k_{\mathrm{max}} <$ 0.1 s/km',
            # r'$A_{\mathrm{noise}}$',
            # '$u_{0}-T_{0}$ + ${R_{s}}^{cdm}$ ($u_{0}$)',
            'Fiducial',
            r'Patchy + $\tau_{\mathrm{CMB}}$ + ${R_{s}}^{cdm}$ ($u_{0}$)']


# Define shapes for each analysis
shapes = ['D', 'D', 'D', 'D', 'D']

# Generate different tones of blue for all analyses except the last one
# colors = cm.Blues(np.linspace(0.3, 0.8, len(analysis) - 1)) #paper
colors = cm.Blues(np.linspace(0.5, 0.8, len(analysis) - 1)) #seminar

# Ensure colors is a 2D array before stacking
colors = np.array(colors)
print(np.shape(colors))

# Add an orange color for the last analysis
colors = np.vstack([colors, [1, 0.5, 0, 1]])
print(np.shape(colors))

print(len(colors))  # Should print 5
print(len(t0_dicts))  # Should print 5

markersize = 7  


# Loop through each t0_dict and plot
for i, t0_dict in enumerate(t0_dicts):
    
    x_values_i = [x + diff * (i) for x in x_values]

    y_values = [t0_dict['t0z4.2']['value'], t0_dict['t0z4.6']['value'], t0_dict['t0z5.0']['value']]
    yerr_lower = [t0_dict['t0z4.2']['lower_error'], t0_dict['t0z4.6']['lower_error'], t0_dict['t0z5.0']['lower_error']]
    yerr_upper = [t0_dict['t0z4.2']['upper_error'], t0_dict['t0z4.6']['upper_error'], t0_dict['t0z5.0']['upper_error']]

    # Adjust yerr arrays to handle cases where lower error is 0
    yerr_lower_adjusted = [0 if err == 0 else err for err in yerr_lower]
    yerr_upper_adjusted = [upper if lower != 0 else upper + lower for lower, upper in zip(yerr_lower, yerr_upper)]

    # Create asymmetric error bars
    yerr = [yerr_lower_adjusted, yerr_upper_adjusted]
    plt.errorbar(x_values_i, y_values, yerr=yerr, fmt=shapes[i], label=analysis[i], color=colors[i], markersize = markersize)


#Eilers+2018:
data = [
    (4.0, 0.9049, 0.0372),
    (4.25, 1.0230, 0.0311),
    (4.5, 1.2286, 0.0651),
    (4.75, 1.6381, 0.0770),
    (5.0, 2.0818, 0.1060),
    (5.25, 2.5321, 0.0984),
    (5.5, 2.9357, 0.1090),
    (5.75, 4.0057, 0.2400)
]

# Extract redshift, effective optical depth, and uncertainties from the data
redshifts_eilers = np.array([entry[0] for entry in data])
effective_optical_depth_eilers = np.array([entry[1] for entry in data])
optical_depth_uncertainty_eilers = np.array([entry[2] for entry in data])
plt.fill_between(redshifts_eilers, effective_optical_depth_eilers - optical_depth_uncertainty_eilers, effective_optical_depth_eilers + optical_depth_uncertainty_eilers, label = 'Eilers+18', color='green', alpha=0.2)

#Becker+2013
data = [
    (2.15, 0.8806, 0.0103),
    (2.25, 0.8590, 0.0098),
    (2.35, 0.8304, 0.0093),
    (2.45, 0.7968, 0.0089),
    (2.55, 0.7810, 0.0090),
    (2.65, 0.7545, 0.0088),
    (2.75, 0.7371, 0.0088),
    (2.85, 0.7167, 0.0086),
    (2.95, 0.6966, 0.0084),
    (3.05, 0.6670, 0.0082),
    (3.15, 0.6385, 0.0080),
    (3.25, 0.6031, 0.0079),
    (3.35, 0.5762, 0.0074),
    (3.45, 0.5548, 0.0071),
    (3.55, 0.5325, 0.0071),
    (3.65, 0.4992, 0.0069),
    (3.75, 0.4723, 0.0068),
    (3.85, 0.4470, 0.0072),
    (3.95, 0.4255, 0.0071),
    (4.05, 0.4030, 0.0071),
    (4.15, 0.3744, 0.0074),
    (4.25, 0.3593, 0.0075),
    (4.35, 0.3441, 0.0102),
    (4.45, 0.3216, 0.0094),
    (4.55, 0.3009, 0.0104),
    (4.65, 0.2881, 0.0117),
    (4.75, 0.2419, 0.0201),
    (4.85, 0.2225, 0.0151)
]

# Extract redshift and transmitted flux values along with their uncertainties
redshifts_becker = np.array([entry[0] for entry in data])
flux = np.array([entry[1] for entry in data])
flux_uncertainty = np.array([entry[2] for entry in data])

# Calculate the effective optical depth: -np.log(Flux)
effective_optical_depth_becker = -np.log(flux)

# Compute the uncertainties in the effective optical depth
# Propagating the uncertainty for the flux (logarithmic error propagation)
effective_optical_depth_minus_sigma_becker = -np.log(flux + flux_uncertainty)
effective_optical_depth_plus_sigma_becker = -np.log(flux - flux_uncertainty)
plt.fill_between(redshifts_becker,  effective_optical_depth_minus_sigma_becker,effective_optical_depth_plus_sigma_becker, color='magenta', label = 'Becker+13', alpha=0.2)



#Bosman+18

data = [
    (4.8, 0.194, -0.015, 0.018, 15),
    (4.9, 0.171, -0.014, 0.014, 17),
    (5.0, 0.1581, -0.0089, 0.0082, 37),
    (5.1, 0.1428, -0.0054, 0.0068, 48),
    (5.2, 0.1222, -0.0054, 0.0046, 55),
    (5.3, 0.1031, -0.0050, 0.0056, 58),
    (5.4, 0.0801, -0.0048, 0.0061, 64),
    (5.5, 0.0591, -0.0035, 0.0039, 64),
    (5.6, 0.0447, -0.0036, 0.0033, 59),
    (5.7, 0.0256, -0.0029, 0.0031, 51),
    (5.8, 0.0172, -0.0028, 0.0022, 45),
    (5.9, 0.0114, -0.0030, 0.0029, 28),
    (6.0, 0.0089, -0.0029, 0.0033, 19),
    (6.1, 0.0088, -0.0074, 0.0082, 10),
    (6.2, 0.0047, -0.0044, 0.0045, 8)
]

# Extract values
redshifts = np.array([entry[0] for entry in data])
FLy_alpha = np.array([entry[1] for entry in data])
FLy_alpha_minus_sigma = np.array([entry[2] for entry in data])
FLy_alpha_plus_sigma = np.array([entry[3] for entry in data])

# Compute effective optical depth tau
effective_optical_depth = -np.log(FLy_alpha)

# Compute uncertainties in tau using sigma_tau = sigma_F / F
sigma_tau_minus = np.abs(FLy_alpha_minus_sigma) / FLy_alpha
sigma_tau_plus = FLy_alpha_plus_sigma / FLy_alpha
plt.fill_between(redshifts, effective_optical_depth - sigma_tau_minus, sigma_tau_plus+ effective_optical_depth, color = 'red', alpha = 0.3, label = 'Bosman+22')


#Boera+19
z = np.array([4.2, 4.6, 5.0])  # Redshift values
tau_eff = np.array([1.02, 1.41, 1.69])  # Central values of τ_eff
tau_err_lower = np.array([0.04, 0.08, 0.10])  # Lower error bars
tau_err_upper = np.array([0.04, 0.09, 0.11])  # Upper error bars

# Compute upper and lower bounds for the shaded region
tau_upper = tau_eff + tau_err_upper
tau_lower = tau_eff - tau_err_lower
x_values_boera = [x + 3*diff for x in x_values]
# Plot data points
# plt.errorbar(x_values_boera, tau_eff, yerr=[tau_err_lower, tau_err_upper], fmt='o', markersize = markersize ,
#              color='purple', label=r'Boera+19') #paper
plt.errorbar(x_values_boera, tau_eff, yerr=[tau_err_lower, tau_err_upper], fmt='o', markersize = markersize ,
             color='gray', label=r'Boera+19') #seminar


#Irsic+24
# Data
z = np.array([4.2, 4.6, 5.0])  # Redshift values
values = np.array([1.09055251, 1.40635851, 1.6570012])  # Central values
lower_bounds = np.array([1.07137092, 1.37792626, 1.60911911])  # Lower bounds
upper_bounds = np.array([1.18973088, 1.4958086, 1.77836617])  # Upper bounds

# Compute errors
lower_errors = values - lower_bounds
upper_errors = upper_bounds - values

# Plot with error bars
x_values_irsic = [x + 4*diff for x in x_values]
# plt.errorbar(x_values_irsic, values, yerr=[lower_errors, upper_errors], fmt='*', 
#              color='saddlebrown',label='Irsic+24', markersize = 9) #paper

plt.errorbar(x_values_irsic, values, yerr=[lower_errors, upper_errors], fmt='*', 
             color='gray',label='Irsic+24', markersize = 11) #seminar


handles, labels = plt.gca().get_legend_handles_labels()
# First legend (top right)
first_legend = plt.legend(handles[:3], labels[:3], frameon=False, loc='upper right')
plt.gca().add_artist(first_legend)

# Second legend (bottom left)
second_legend = plt.legend(handles[3:], labels[3:], frameon=False, loc='upper left')
plt.gca().add_artist(second_legend)

# # Third legend (top left)
# third_legend = plt.legend(handles[6:], labels[6:], frameon=False, loc='lower right')
# plt.gca().add_artist(third_legend)
plt.ylim(0.7,3.5)
plt.ylabel(rf'$\tau_{{\mathrm{{eff}}}}$')
plt.xlabel('Redshift $z$')
plt.xlim(3.8, 5.5)
plt.xticks([4, 4.5, 5])
plt.savefig('/data/og313/lyalpha/code_wdm/seminar/taueff_evol_test3.pdf', bbox_inches='tight')
plt.show()
# %%
