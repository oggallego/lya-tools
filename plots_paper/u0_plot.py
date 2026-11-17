


#%%
from funcs import get_samples_smooth
import numpy as np
from getdist import plots
from cobaya import load_samples
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.cm as cm

#Plot evolution for u0(z)
import sys
sys.path.append('/data/emergence11/irsic/thermal_history')
import RT1d_model as RT
from matplotlib import cm


from funcs import get_samples_irsic, get_bf_sigma_irsic_all, get_sigma_irsic_alpha
name_chain= '/data/og313/lyalpha/code_wdm/chains_irsic/chain_bo_WDM_pRELICS_T0G_moC'
samples_irsic = get_samples_irsic(name_chain)
bestfit, onesigma_all = get_bf_sigma_irsic_all(name_chain, samples_irsic)
one_sigma, two_sigma =  get_sigma_irsic_alpha(samples_irsic)


def get_u0_bf(name):
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
    print('Best-fit parameters for u0 are', arr[9], arr[10], arr[11])
    u0z42 = arr[9]
    u0z46 = arr[10]
    u0z5= arr[11]
    
    onesigma = []
    u0z_params = ['u0z42', 'u0z46', 'u0z5']
    for i in u0z_params:
        lims = stats.parWithName(f'{i}').limits
        for conf, lims in zip(samples.contours,lims):
            if conf == 0.68:
                print((lims.lower, lims.upper))
                onesigma.append((lims.lower, lims.upper))  # Append as a tuple

    # Get 1 sigma error bars:
    lower_error_u0z42 = u0z42 - onesigma[0][0]
    upper_error_u0z42 = onesigma[0][1] - u0z42
    
    lower_error_u0z46 = u0z46 - onesigma[1][0]
    upper_error_u0z46 = onesigma[1][1] - u0z46

    lower_error_u0z5 = u0z5 - onesigma[2][0]
    upper_error_u0z5 = onesigma[2][1] - u0z5

    if np.any(np.array([lower_error_u0z42, upper_error_u0z42]) < 0):
        print(lower_error_u0z42, upper_error_u0z42, 'One is negative')
        onesigma = []
        u0z_param = ['u0z42']
        sigma = 1-0.682689492137086

        ### more accurately
        two_sigma = 1-0.954499736103642
        three_sigma = 1-0.997300203936740
        four_sigma = 1-0.999936657516334# = 4sigma
        five_sigma = 1-0.999999426696856 #= 5sigma

        upper_bound_sig = (samples.confidence('u0z42', limfrac=sigma, upper=True))
        upper_bound_twosig = (samples.confidence('u0z42', limfrac=two_sigma, upper=True))
        print('u0z42 <', upper_bound_sig)
        upper_error_u0z42 = upper_bound_sig - u0z42
        lower_error_u0z42 = 0
        print('New upper bound', upper_error_u0z42)


        # Check for u0z4.6
    if np.any(np.array([lower_error_u0z46, upper_error_u0z46]) < 0):
        print(lower_error_u0z46, upper_error_u0z46, 'One is negative')
        onesigma = []
        u0z_param = ['u0z46']
        sigma = 1 - 0.682689492137086

        ### more accurately
        two_sigma = 1 - 0.954499736103642
        three_sigma = 1 - 0.997300203936740
        four_sigma = 1 - 0.999936657516334  # = 4sigma
        five_sigma = 1 - 0.999999426696856  # = 5sigma

        upper_bound_sig = samples.confidence('u0z46', limfrac=sigma, upper=True)
        upper_bound_twosig = samples.confidence('u0z46', limfrac=two_sigma, upper=True)
        print('u0z46 <', upper_bound_sig)
        upper_error_u0z46 = upper_bound_sig - u0z46
        lower_error_u0z46 =  0
        print('New upper bound', upper_error_u0z46)

    # Check for u0z5.0
    if np.any(np.array([lower_error_u0z5, upper_error_u0z5]) < 0):
        print(lower_error_u0z5, upper_error_u0z5, 'One is negative')
        onesigma = []
        u0z_param = ['u0z5']
        sigma = 1 - 0.682689492137086

        ### more accurately
        two_sigma = 1 - 0.954499736103642
        three_sigma = 1 - 0.997300203936740
        four_sigma = 1 - 0.999936657516334  # = 4sigma
        five_sigma = 1 - 0.999999426696856  # = 5sigma

        upper_bound_sig = samples.confidence('u0z5', limfrac=sigma, upper=True)
        upper_bound_twosig = samples.confidence('u0z5', limfrac=two_sigma, upper=True)
        print('u0z5 <', upper_bound_sig)
        upper_error_u0z5 = upper_bound_sig - u0z5
        lower_error_u0z5 = 0
        print('New upper bound', upper_error_u0z5)

    # Dict to get results:
    u0_dict = {
        'u0z4.2': {
            'value': u0z42,
            'lower_error': lower_error_u0z42,
            'upper_error': upper_error_u0z42
        },
        'u0z4.6': {
            'value': u0z46,
            'lower_error': lower_error_u0z46,
            'upper_error': upper_error_u0z46
        },
        'u0z5.0': {
            'value': u0z5,
            'lower_error': lower_error_u0z5,
            'upper_error': upper_error_u0z5
        }
    }
    # Print information for each entry
    for key, value in u0_dict.items():
        print(f"For {key}:")
        print(f"  Best fit value: {value['value']}")
        print(f"  Lower error: {value['lower_error']}")
        print(f"  Upper error: {value['upper_error']}")
        print()
    
    return u0_dict


markersize = 5
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

fontsize = 20
plt.rcParams["font.size"] = fontsize   # Global font size
plt.rcParams["axes.labelsize"] = fontsize  # Axis labels
plt.rcParams["xtick.labelsize"] = fontsize   # X-axis tick labels
plt.rcParams["ytick.labelsize"] = fontsize   # Y-axis tick labels
plt.rcParams["legend.fontsize"] = 20   # Legend text
plt.figure(figsize = (12, 9))

# Define a list of u0_dicts for different analyses
u0_dicts = [
    get_u0_bf('chain_fwdm_not0g_Rscdm_kmax'),
    get_u0_bf('/noise_runs/chain_fwdm_noise_lognorm_start'),
    get_u0_bf('chain_fwdm_u0t0_t0g_lowcut_Rsu0_post'),
    get_u0_bf('chain_fwdm_fiducial'),
    get_u0_bf('chain_fwdm_repatchy_Rsu0_post_tau')
    # Add more analyses as needed
]

analysis = [r'$k_{\mathrm{max}} <$ 0.1 s/km',
            r'$A_{\mathrm{noise}}$',
            r'$u_{0}-T_{0}$ + ${R_{s}}^{\mathrm{cdm}}$ ($u_{0}$)',
            'Fiducial',
            r'Patchy + $\tau_{\mathrm{CMB}}$ + ${R_{s}}^{\mathrm{cdm}}$ ($u_{0}$)']

# Define shapes for each analysis
shapes = ['D', 'D', 'D', 'D', 'D']

colors = cm.Blues(np.linspace(0.3, 0.8, len(analysis) - 1))

# Ensure colors is a 2D array before stacking
colors = np.array(colors)
print(np.shape(colors))

# Add an orange color for the last analysis
colors = np.vstack([colors, [1, 0.5, 0, 1]])
print(np.shape(colors))

print(len(colors))  # Should print 5
print(len(u0_dicts))  # Should print 5

diff = 0.03

# TRY EVOLUTION HERE:
base = '/data/emergence11/irsic/thermal_history/'
# RT3 = RT.RT1d_model('irsic2023.dat',base,readin_only=False)
# ztime1,temp1,u1 = RT3.get_history()
# T03,u03 = RT3.get_u0T0()
# lbl1 ='Irsic+2024'
# plt.plot(ztime1,u1,color='saddlebrown',ls='-',label=lbl1,lw=2)

# lbl1 = 'Villasenor+22'
# RT2 = RT.RT1d_model('villasenor2021.dat',base,readin_only=False)
# ztime1,temp1,u1 = RT2.get_history()
# plt.plot(ztime1, u1,color='r',ls='-',label=lbl1,lw=2)

RT1 = RT.RT1d_model('puchwein2019.dat',base,readin_only=False)
ztime1,temp1,u1 = RT1.get_history()
T01,u01 = RT1.get_u0T0()
plt.plot(ztime1,u1,color='black',ls='-',label='Puchwein+19',lw=2)

markersize = 7
# Plot Boera+19 measurements
x_values = [4.2, 4.6, 5.0]
u0_ref = [7.29, 7.10, 4.57]
du0_high = [0.98, 0.83, 1.37]
du0_low = [1.35, 1.45, 1.16]
x_values_boera = [x + 5*diff for x in x_values]
plt.errorbar(x_values_boera, u0_ref, yerr=[du0_low, du0_high], fmt='*',  label='Boera+19', color='purple', markersize = markersize+3)

# Rogers measurements:
x_values_rogers = [x + 6*diff for x in x_values]
u0_r21_ref=[11.84,12.22,7.26]
du0_r21_high=[5.46,4.98,2.40]
du0_r21_low=[5.46,4.98,2.40]
plt.errorbar(x_values_rogers, u0_r21_ref, yerr=[du0_r21_low, du0_r21_high],  fmt='v', label='Rogers+21',  color = 'tab:green', markersize = markersize)

# Villasenor measurments:
x_values_villa = [x + 7*diff for x in x_values]
u0_v22_ref=[6.0878,5.603,5.2563]

du0_v22_high=-np.array(u0_v22_ref)+np.array([6.91262758,6.41899258,6.05149578]) #0.1*u0_v22_ref                                    
du0_v22_low=np.array(u0_v22_ref)-np.array([4.99655592,4.54443141,4.21603345]) #0.1*u0_v22_ref 
plt.errorbar(x_values_villa, u0_v22_ref, yerr=[du0_v22_low, du0_v22_high],  fmt='^', label='Villasenor+22',  color = 'tab:red',  markersize = markersize)


# Irsic+24:
plt.errorbar(
    [4.2 +8*diff], [bestfit[9]],
     yerr=[[bestfit[9] - onesigma_all[9][0]], [onesigma_all[9][1]-bestfit[9]]],
     fmt='s', label='Irsic+24',  color = 'saddlebrown',  markersize =markersize)

plt.errorbar(
    [4.6+8*diff], [bestfit[10]],
    yerr=[[bestfit[10] - onesigma_all[10][0]], [onesigma_all[10][1]-bestfit[10]]],
    fmt='s', color='saddlebrown',  markersize = markersize)

plt.errorbar(
    [5.0 +8*diff], [bestfit[11]],
    yerr=[[bestfit[11] - onesigma_all[11][0]], [onesigma_all[11][1]-bestfit[11]]],
    fmt='s',color='saddlebrown',  markersize = markersize)
markersize = 5

# Loop through each u0_dict and plot
for i, u0_dict in enumerate(u0_dicts):
    x_values_i = [x + diff * (i) for x in x_values]

    y_values = [u0_dict['u0z4.2']['value'], u0_dict['u0z4.6']['value'], u0_dict['u0z5.0']['value']]
    yerr_lower = [u0_dict['u0z4.2']['lower_error'], u0_dict['u0z4.6']['lower_error'], u0_dict['u0z5.0']['lower_error']]
    yerr_upper = [u0_dict['u0z4.2']['upper_error'], u0_dict['u0z4.6']['upper_error'], u0_dict['u0z5.0']['upper_error']]

    # Adjust yerr arrays to handle cases where lower error is 0
    yerr_lower_adjusted = [0 if err == 0 else err for err in yerr_lower]
    yerr_upper_adjusted = [upper if lower != 0 else upper + lower for lower, upper in zip(yerr_lower, yerr_upper)]

    # Create asymmetric error bars
    yerr = [yerr_lower_adjusted, yerr_upper_adjusted]
    plt.errorbar(x_values_i, y_values, yerr=yerr, fmt=shapes[i], label=analysis[i], color=colors[i])

plt.ylabel(rf'$u_{0}\,$ [eV$\,m_{{p}}^{{-1}}$]')
plt.xlabel(f'Redshift $z$')
plt.ylim(2.4, 15)
plt.yticks([5, 10, 15])
# plt.legend(frameon=False, loc = 'upper right')# bbox_to_anchor=(1.05, 0.9))

# Create the first legend for the first three lines
handles, labels = plt.gca().get_legend_handles_labels()
first_legend = plt.legend(handles[:1], labels[:1], frameon=False, loc='lower right', bbox_to_anchor=(0.9, 0.1)) #bbox_to_anchor=(0.15, 1))

# Add the first legend back to the plot
plt.gca().add_artist(first_legend)

# Create the second legend for the remaining lines
plt.legend(handles[1:], labels[1:], frameon=False, loc='upper right')
plt.xlim(4, 6.5)
# plt.xticks([4.2, 4.6, 5.0])
plt.savefig('/data/og313/lyalpha/code_wdm/paper/draft/u0_evol_test_noise2.pdf',
             bbox_inches='tight')
plt.show()
# %%
