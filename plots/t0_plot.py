
#%%
from funcs import get_samples_smooth
import numpy as np
from getdist import plots
#from cobaya import load_samples
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.cm as cm

#Plot evolution for T0(z)
import sys
sys.path.append('/data/emergence11/irsic/thermal_history')
import RT1d_model as RT
from matplotlib import cm




from funcs import get_samples_irsic, get_bf_sigma_irsic_all, get_sigma_irsic_alpha
name_chain= '/data/vault/og313/chains_irsic/chain_bo_WDM_pRELICS_patchy_T0G_mru0_moC'
samples_irsic = get_samples_irsic(name_chain)
bestfit, onesigma_all = get_bf_sigma_irsic_all(name_chain, samples_irsic)
one_sigma, two_sigma =  get_sigma_irsic_alpha(samples_irsic)

def get_t0_bf(name):
    directory = f'/data/og313/lyalpha/code_wdm/archive_chains/{name}'
    burning = 0.8
    samples = get_samples_smooth(directory, burning)
    # stats = samples.getLikeStats()
    stats = samples.getMargeStats()

    if 'noise' in name:
        n_pars = 17
    else:
        n_pars = 14

    for i in range(n_pars):
        param = stats.parWithName(f'{i}')
        if param is not None:
            lims = param.limits
            # Process limits as needed
        else:
            print(f"Parameter {i} not found in stats")
    
    # arr = [stats.names[i].bestfit_sample for i in range(n_pars)]
    arr = [stats.names[i].mean for i in range(n_pars)]
    print('Best-fit parameters for u0 are', arr[0], arr[1], arr[2])
    t0z42 = arr[0]
    t0z46 = arr[1]
    t0z5= arr[2]
    
    onesigma = []
    t0z_params = ['T0z42', 'T0z46', 'T0z5']
    for i in t0z_params:
        lims = stats.parWithName(f'{i}').limits
        for conf, lims in zip(samples.contours,lims):
            if conf == 0.68:
                print((lims.lower, lims.upper))
                onesigma.append((lims.lower, lims.upper))  # Append as a tuple

    # Get 1 sigma error bars:
    # Get 1 sigma error bars:
    lower_error_t0z42 = max(t0z42 - onesigma[0][0], 0)
    upper_error_t0z42 = max(onesigma[0][1] - t0z42, 0)

    lower_error_t0z46 = max(t0z46 - onesigma[1][0], 0)
    upper_error_t0z46 = max(onesigma[1][1] - t0z46, 0)

    lower_error_t0z5 = max(t0z5 - onesigma[2][0], 0)
    upper_error_t0z5 = max(onesigma[2][1] - t0z5, 0)

    # Put instead 2 sigma upper lims:
    # if 'kmax' in name:
    #     upper_error_t0z42 = 0
    #     upper_error_t0z46 = 0
    #     upper_error_t0z5 = 0

    # Dict to get results:
    t0_dict = {
        't0z4.2': {
            'value': t0z42,
            'lower_error': lower_error_t0z42,
            'upper_error': upper_error_t0z42
        },
        't0z4.6': {
            'value': t0z46,
            'lower_error': lower_error_t0z46,
            'upper_error': upper_error_t0z46
        },
        't0z5.0': {
            'value': t0z5,
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


markersize = 10
fontsize = 28
plt.rcParams["font.size"] = fontsize   # Global font size
plt.rcParams["axes.labelsize"] = fontsize  # Axis labels
plt.rcParams["xtick.labelsize"] = fontsize   # X-axis tick labels
plt.rcParams["ytick.labelsize"] = fontsize   # Y-axis tick labels
plt.rcParams["legend.fontsize"] = 27   # Legend text
plt.figure(figsize = (16, 11))

x_values = [4.2, 4.6, 5.0]
diff = 0.04
t0_dicts = [
    get_t0_bf('chain_fwdm_not0g_Rscdm_kmax'),
    # get_t0_bf('/noise_runs/chain_fwdm_noise_lognorm'),
    # get_t0_bf('chain_fwdm_u0t0_t0g_lowcut_Rsu0_post'),
    get_t0_bf('chain_fwdm_fiducial'),
    get_t0_bf('chain_fwdm_repatchy_Rsu0_post_tau')
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
colors = cm.Blues(np.linspace(0.5, 0.8, len(analysis) - 1))

# Ensure colors is a 2D array before stacking
colors = np.array(colors)
print(np.shape(colors))

# Add an orange color for the last analysis
colors = np.vstack([colors, [1, 0.5, 0, 1]])
print(np.shape(colors))

print(len(colors))  # Should print 5
print(len(t0_dicts))  # Should print 5


base = '/data/emergence11/irsic/thermal_history/'
RT1 = RT.RT1d_model('puchwein2019.dat',base,readin_only=True)
ztime1,temp1,u1 = RT1.get_history()
T01,u01 = RT1.get_u0T0()
plt.plot(ztime1,temp1/1.0e4,color='black',ls='-',label='Puchwein+19',lw=2)

# lbl1 = 'Villasenor+22'
# RT2 = RT.RT1d_model('villasenor2021.dat',base,readin_only=True)
# ztime1,temp1,u1 = RT2.get_history()
# plt.plot(ztime1,temp1/1.0e4,color='r',ls='-',label=lbl1,lw=2)

# RT3 = RT.RT1d_model('irsic2023.dat',base,readin_only=True)
# ztime1,temp1,u1 = RT3.get_history()
# T03,u03 = RT3.get_u0T0()
# lbl1 ='Irsic+2024'
# plt.plot(ztime1,temp1/1.0e4,color='saddlebrown',ls='-',label=lbl1,lw=2)
# zarr=np.array([4.2,4.6,5.0])
# sigma = 1e3
# for i,t in enumerate(T03):
#     plt.errorbar(zarr[i],T03[i]/1e4,sigma/1e4,marker='o',ls='',
#                  color='saddlebrown',fmt = 'o', markersize = markersize)

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


print('y upper', yerr_upper)
# print('ylower', yerr_lower)
# Boera+19 measurements:
T0_r21_ref=[1.089,1.1330,1.0524]
dT0_r21_high=[0.155,0.151,0.1329]
dT0_r21_low=[0.155,0.151,0.1329]
x_values_rogers = [x + 3*diff for x in x_values]
# plt.errorbar(x_values_rogers, T0_r21_ref, yerr=[dT0_r21_low, dT0_r21_high],  fmt='v', label='Rogers+21', color = 'tab:green', markersize = markersize) #paper
plt.errorbar(x_values_rogers, T0_r21_ref, yerr=[dT0_r21_low, dT0_r21_high],  fmt='v', label='Rogers+21', color = 'gray', markersize = markersize) #seminar

# Villasenor measurements:
T0_v22_ref=[0.9073,0.8924,0.9350]
dT0_v22_high=0.1 * np.array(T0_v22_ref) #-T0_v22_ref+[0.956042700019,0.942916828264,0.986933590842] #0.1 * T0_v22_ref    
dT0_v22_low=0.1 * np.array(T0_v22_ref)
x_values_villa = [x + 4*diff for x in x_values]
# plt.errorbar(x_values_villa, T0_v22_ref, yerr=[dT0_v22_low, dT0_v22_high],  fmt='s', label='Villasenor+22', color = 'tab:red', markersize = markersize) #paper
plt.errorbar(x_values_villa, T0_v22_ref, yerr=[dT0_v22_low, dT0_v22_high],  fmt='s', label='Villasenor+22', color = 'gray', markersize = markersize) #seminar


#Boera+19:
x_values_boera = [x + 5*diff for x in x_values]
T0_ref=[0.813,0.731,0.737]
dT0_high=[0.134,0.135,0.167]
dT0_low=[0.097,0.088,0.139]
# plt.errorbar(x_values_boera, T0_ref, yerr=[dT0_low, dT0_high], fmt='o',  label='Boera+19', color = 'purple', markersize = markersize) #paper
plt.errorbar(x_values_boera, T0_ref, yerr=[dT0_low, dT0_high], fmt='o',  label='Boera+19', color = 'gray', markersize = markersize) #semina

#Rogers+21:
T0_r21_ref=[1.089,1.1330,1.0524]
dT0_r21_high=[0.155,0.151,0.1329]
dT0_r21_low=[0.155,0.151,0.1329]

# Irsic+24:
# plt.errorbar(
#     [4.2+8*diff], [bestfit[0]],
#      yerr=[[bestfit[0] - onesigma_all[0][0]], [onesigma_all[0][1]-bestfit[0]]],
#     fmt = '*', markersize = 11, label = 'Irsic+24', color = 'saddlebrown') #paper
plt.errorbar(
    [4.2+6*diff], [bestfit[0]],
     yerr=[[bestfit[0] - onesigma_all[0][0]], [onesigma_all[0][1]-bestfit[0]]],
    fmt = '*', markersize = 13, label = 'Irsic+24', color = 'gray') #seminar

# plt.errorbar(
#         [4.6+8*diff], [bestfit[1]],
#        yerr=[[bestfit[1] - onesigma_all[1][0]], [onesigma_all[1][1]-bestfit[1]]],
#         fmt = '*', markersize = 11, color = 'saddlebrown') #paper
plt.errorbar(
        [4.6+6*diff], [bestfit[1]],
       yerr=[[bestfit[1] - onesigma_all[1][0]], [onesigma_all[1][1]-bestfit[1]]],
        fmt = '*', markersize = 13, color = 'gray') #seminar

# plt.errorbar(
#         [5.0+8*diff], [bestfit[2]],
#        yerr=[[bestfit[2] - onesigma_all[2][0]], [onesigma_all[2][1]-bestfit[2]]],
#         fmt = '*', markersize = 11,  color = 'saddlebrown') #paper
plt.errorbar(
        [5.0+6*diff], [bestfit[2]],
       yerr=[[bestfit[2] - onesigma_all[2][0]], [onesigma_all[2][1]-bestfit[2]]],
        fmt = '*', markersize = 13,  color = 'gray') #seminar


#Gaikwad:20:
T0 = [1.1, 1.05, 1.2]
yerr_lower = [0.1600, 0.2100, 0.2200]
yerr_upper = [0.1600, 0.2100, 0.2200]
z_values = [5.4, 5.6, 5.8]
T0_lower = np.array(T0) - np.array(yerr_lower)
T0_upper = np.array(T0) + np.array(yerr_upper)
# Plot the filled band
# plt.fill_between(z_values, T0_lower, T0_upper, color='gray', alpha=0.5, label='Gaikwad+20') #paper
plt.fill_between(z_values, T0_lower, T0_upper, color='green', alpha=0.3 , label='Gaikwad+20') #seminar
#plt.plot(z_values, T0, 'o', color='gray', label='T0 Values')


# Gaikwad+21:
z_values = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8]
dz = 0.1
T0 = [0.9500, 1.1000, 1.2750, 1.3500, 1.4750, 1.4750, 1.2750, 1.1250, 1.0250, 0.9250]
dT0 = [0.1393, 0.1028, 0.1132, 0.1390, 0.1341, 0.1322, 0.1493, 0.1125, 0.1070, 0.0876]

# Calculate the upper and lower bounds of the uncertainties
T0_lower = np.array(T0) - np.array(dT0)
T0_upper = np.array(T0) + np.array(dT0)

# Plot the filled band
plt.fill_between(z_values, T0_lower, T0_upper, color='magenta', alpha=0.2, label='Gaikwad+20b')
#plt.plot(z_values, T0, 'o', color='tab:green', label='T0 Values')

# Puchwein+19:
# Read by eye but ask Vid:
# T0 = [1.1, 1.45, 1.62, 1.35, 1.1, 1.0, 0.98, 1, 1.1, 1.2]
# z_values = [2, 2.5, 3, 3.5, 4, 4.4, 4.75, 5, 5.5,6]
# plt.plot(z_values, T0, label = 'Puchwein+19', color = 'black')




handles, labels = plt.gca().get_legend_handles_labels()
# First legend (top right)
first_legend = plt.legend(handles[:3], labels[:3], frameon=False, loc='upper right')
plt.gca().add_artist(first_legend)

# Second legend (bottom left)
second_legend = plt.legend(handles[3:], labels[3:], frameon=False, loc='lower left')
plt.gca().add_artist(second_legend)

# # Third legend (top left)
# third_legend = plt.legend(handles[6:], labels[6:], frameon=False, loc='lower right')
# plt.gca().add_artist(third_legend)

plt.ylabel(rf'$T_{0}\,/\,10^{4}$ (K)')
plt.xlabel('Redshift $z$')
plt.xlim(2, 6)
plt.xticks([2,3,4,5,6])
plt.savefig('/data/og313/lyalpha/code_wdm/seminar/t0_evol_test3.pdf', bbox_inches='tight')
plt.show()
# %%

# %%

# %%
