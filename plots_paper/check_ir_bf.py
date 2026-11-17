#%%
import numpy as np
import sys
sys.path.append('/data/og313/lyalpha/code_wdm/nn_emulator_fwdm')
from cobaya_analysis import compute_chi_squared
sys.path.append('/data/og313/lyalpha/code_wdm/mcmc')
from get_boera_data import get_data
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt, ticker as mticker
import matplotlib.ticker as ticker
log_k, P_kRC42, P_kRC46, P_kRC5, covariance_matrix = get_data()


pk42 = np.loadtxt('/data/og313/lyalpha/code_wdm/code_plots/pk_bestfit_irsic2024/pk_bo_WDM_pRELICS_T0G_mru0_moC.dat')[:16, 2]
pk46 = np.loadtxt('/data/og313/lyalpha/code_wdm/code_plots/pk_bestfit_irsic2024/pk_bo_WDM_pRELICS_T0G_mru0_moC.dat')[16:32, 2]
pk5 = np.loadtxt('/data/og313/lyalpha/code_wdm/code_plots/pk_bestfit_irsic2024/pk_bo_WDM_pRELICS_T0G_mru0_moC.dat')[32:48, 2]
print((pk42))
k = 10**log_k
print(P_kRC42)
# residuals = pk5- P_kRC5
# cov_matrix = covariance_matrix[32:48, 32:48]

# print('Residuals', residuals)

# def compute_chi_squared(residuals, cov_matrix):
#     # Compute the inverse of the covariance matrix
#     cov_matrix_inv = np.linalg.inv(cov_matrix)
    
#     # Compute the chi-squared value
#     chi2 = np.dot(residuals.T, np.dot(cov_matrix_inv, residuals))
    
#     return chi2

# chi46_value = compute_chi_squared(residuals.reshape(-1,1), cov_matrix)
# print(f"Chi-squared: {chi46_value}") 


# plt.rcParams.update({'font.size': 15})

# # Create the figure
# # ~


fig, ax = plt.subplots(1, 3, figsize=(15, 5))
fig.subplots_adjust(wspace=0.4) 
%matplotlib inline

logk = np.arange(-2.2,-0.7, 0.1)
k = 10**logk

ax[0].plot(log_k, pk42, linestyle='--',  label=f'z = 4.2, Irsic+24', markersize=6,color='tab:blue')
ax[0].errorbar(log_k, P_kRC42 * k / np.pi, label='Boera+19', yerr=(k/ np.pi)*np.sqrt(covariance_matrix[:16, :16].diagonal()), fmt='D', markersize=6, color='tab:blue')
ax[0].set_ylabel('kP(k)$/\pi$')
ax[0].set_xlabel('$\mathrm{log}_{10}$ ($k$ $\mathrm{[km^{-1}\,s])}$')
ax[0].set_ylim(0.01,0.8)
ax[0].set_xlim(-2.25, -0.65)
ax[0].set_yscale('log')
ax[0].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax[0].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax[0].legend()


# Repeat similar setup for z = 4.6
ax[1].plot(log_k, pk46* k / np.pi, linestyle='--', color='tab:orange')

# ax[1].plot(log_k, bf_pred_vid_46* k / np.pi, linestyle='--', color='red', label=f'$\chi^{{2}} = {round(chi2_46_vid, 2)}$')
ax[1].errorbar(log_k, P_kRC46 * k / np.pi,  yerr=(k/ np.pi)*np.sqrt(covariance_matrix[16:32, 16:32].diagonal()), fmt='D', markersize=6, color='tab:orange')
ax[1].set_ylabel('kP(k)$/\pi$')
ax[1].set_xlabel('$\mathrm{log}_{10}$ ($k$ $\mathrm{[km^{-1}\,s])}$')
ax[1].set_ylim(0.01,0.8)
ax[1].set_xlim(-2.25, -0.65)
ax[1].set_yscale('log')
ax[1].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax[1].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax[1].legend()


# Repeat similar setup for z = 5.0
ax[2].plot(log_k, pk5*k/np.pi, linestyle='--', color='tab:green', label=f'z = 5.0')

ax[2].errorbar(log_k, P_kRC5 * k / np.pi,  yerr=(k/ np.pi)*np.sqrt(covariance_matrix[32:, 32:].diagonal()), fmt='D', markersize=6, color='tab:green')

ax[2].set_ylabel('kP(k)$/\pi$')
ax[2].set_xlabel('$\mathrm{log}_{10}$ ($k$ $\mathrm{[km^{-1}\,s])}$')
ax[2].set_ylim(0.01,0.8)
ax[2].set_xlim(-2.25, -0.65)
ax[2].set_yscale('log')
ax[2].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax[2].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax[2].legend()



save_path = None

#fig.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
#~fig.suptitle('Common Title', fontsize=16)
if save_path != None:
    plt.savefig(save_path, bbox_inches='tight') 
plt.subplots_adjust(right=0.95)
plt.show()

# plt.plot(log_k, k*pk5/np.pi, linestyle='--', color='tab:green', label=f'z = 5.0')

# # plt.scatter(log_k, (10**log_k)*P_kRC42/np.pi, label = 'Boera+19')
# plt.plot(log_k, (10**log_k)*pk42/np.pi)
# plt.yscale('log')
# plt.show()

# %%

# %%
