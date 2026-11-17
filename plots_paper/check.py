#%%
from funcs import get_samples_smooth, get_best_fit, get_models_z
import numpy as np
import sys
sys.path.append('/data/og313/lyalpha/code_wdm/mcmc')
from get_boera_data import get_data
import sys
sys.path.append('/data/og313/lyalpha/code_wdm/nn_emulator_fwdm')
from model import NNInterpolator
from predictor import get_nn_prediction_mcmc, get_nn_prediction_post_rs_mcmc
from cobaya_analysis import compute_chi_squared
import pandas as pd
import matplotlib.pyplot as plt
# Fiducial analysis:
# directory = '/data/og313/lyalpha/code_wdm/archive_chains/chain_fwdm_fiducial'
# burning = 0.8
# samples = get_samples_smooth(directory, burning)

base_path = '/data/og313/lyalpha/code_wdm/analysis_no_RC'
# chi42_value, chi46_value, chi5_value, bf_pred_42, bf_pred_46, bf_pred_5 = get_best_fit(samples, base_path, post=False)

# Analysis wrt we are comparing:
# name = 'chain_fwdm_fiducial_Rsu0_post'
# directory =f'/data/og313/lyalpha/code_wdm/archive_chains/{name}'
# burning = 0.9
# samples2 = get_samples_smooth(directory, burning)


# chi42_value2, chi46_value2, chi5_value2, bf_pred_42_2, bf_pred_46_2, bf_pred_5_2 = get_best_fit(samples2, base_path, post=True)


# Vid's analysis:
# Get chains from Vid and overplot them

from getdist import loadMCSamples
name_chain = 'chain_bo_WDM_pRELICS_T0G_mru0_moC'

directory = f'/data/emergence9/og313/vid_chains/{name_chain}'

smooth_2D = 0.5
smooth_1D = 0.5
ignore_rows = 0.8

# Orden of names in Vid's saved chains
original_names = ['taueffz42','taueffz46', 'taueffz5', 'T0z42', 'T0z46', 'T0z5',
                'gammaz42', 'gammaz46', 'gammaz5', 
                'u0z42', 'u0z46', 'u0z5',
                'sigma8', 'neff', 'alpha', 'fqso']

# Load the samples without altering parameter names yet
# To run this need .chainparamnames in the same folder
samples_vid = loadMCSamples(directory, settings={
    'ignore_rows': ignore_rows,
    'smooth_scale_2D': smooth_2D,
    'smooth_scale_1D': smooth_1D
})

# # Set the original parameter names
samples_vid.setParamNames(original_names)

names =  [
    'T0z42', 'T0z46', 'T0z5',
    'gammaz42', 'gammaz46', 'gammaz5',
    'taueffz42', 'taueffz46', 'taueffz5',
    'u0z42', 'u0z46', 'u0z5',
    'alpha'
]

    
param_ranges = {'T0z42': (0.5, 1.4),
'T0z46': (0.5, 1.4),
'T0z5': (0.5, 1.4),
'gammaz42': (0.9, 1.8),
'gammaz46': (0.9, 1.8),
'gammaz5': (0.9, 1.8),
'taueffz42': (0.31303136056832276, 1.773844376553829),
'taueffz46': (0.42104312690706497, 2.3859110524733684),
'taueffz5': (0.554854964068882, 3.1441781297236644),
'u0z42': (4.0239696, 21.106591),
'u0z46': (3.6469614, 21.078213),
'u0z5': (2.4634363, 18.783147),
'alpha': (0, 1e+16)}
# Set the ranges for the parameters
samples_vid.setRanges(param_ranges)
from getdist import plots

# %matplotlib inline
# g = plots.get_single_plotter(width_inch=8)
# g.plot_1d(samples_vid, 'alpha', title_limit=2, ax=0, colors=['r', 'g', 'b'])

# sigma = 1-0.682689492137086
# # Get sigma errors on other parameters:
# two_sigma = 1-0.954499736103642                                                                    #
# three_sigma = 1-0.997300203936740
# four_sigma = 1-0.999936657516334# = 4sigma
# five_sigma = 1-0.999999426696856 #= 5sigma

# upper_bound_sig = (samples_vid.confidence('alpha', limfrac=sigma, upper=True))
# upper_bound_twosig = (samples_vid.confidence('alpha', limfrac=two_sigma, upper=True))
# print('mwdm <', 1/upper_bound_twosig, 'keV  is excluded at 2 sigman on Irsic+24')
stats = samples_vid.getMargeStats()
# lims0 = stats.parWithName('T0z42').limits
# lims1 = stats.parWithName('T0z46').limits
# for conf, lim0, lim1 in zip(samples_vid.contours,lims0, lims1):
#     print('x0 %s%% lower: %.3f upper: %.3f (%s)'%(conf, lim0.lower, lim0.upper, lim0.limitType()))
#     print('x1 %s%% lower: %.3f upper: %.3f (%s)'%(conf, lim1.lower, lim1.upper, lim1.limitType()))

# # Get one sigma for each param
# onesigma_all = []
# for i in names:

#     lims = stats.parWithName(f'{i}').limits
#     for conf, lims in zip(samples_vid.contours,lims):
#         # (conf, lim0.lower, lim0.upper,
#         # print(f'Parameter {i}: {conf}, {lims.lower} and {lims.upper}')
#         if conf == 0.68:
#             onesigma_all.append((lims.lower, lims.upper))
#             # [0] T0z42, [1] T0 z46 ...
# # Get bestfit values:
# bestfit = np.zeros(len(original_names))
# result = samples_vid.getLikeStats()
# for i in range(len(bestfit)):
#     bestfit[i] = result.names[i].bestfit_sample
#     print(original_names[i], bestfit[i])

# bestfit_arr = [bestfit[3], bestfit[4], bestfit[5], bestfit[6], bestfit[7],
# bestfit[8], bestfit[0], bestfit[1], bestfit[2], bestfit[9], bestfit[10],
# bestfit[11], bestfit[-2]]
# print(bestfit_arr)

samples_irsic = samples_vid
stats = samples_irsic.getMargeStats()
names =  [
'T0z42', 'T0z46', 'T0z5',
'gammaz42', 'gammaz46', 'gammaz5',
'taueffz42', 'taueffz46', 'taueffz5',
'u0z42', 'u0z46', 'u0z5',
'alpha'
]
onesigma_all = []
for i in names:

    lims = stats.parWithName(f'{i}').limits
    for conf, lims in zip(samples_irsic.contours,lims):
        # (conf, lim0.lower, lim0.upper,
        # print(f'Parameter {i}: {conf}, {lims.lower} and {lims.upper}')
        if conf == 0.68:
            onesigma_all.append((lims.lower, lims.upper))
        # [0] T0z42, [1] T0 z46 ...


bestfit = np.zeros(len(names))
result = samples_irsic.getLikeStats()
for i in range(len(bestfit)):
    bestfit[i] = result.names[i].bestfit_sample

bestfit_arr = np.array([bestfit[3], bestfit[4], bestfit[5], bestfit[6], bestfit[7],
bestfit[8], bestfit[0], bestfit[1], bestfit[2], bestfit[9], bestfit[10],
bestfit[11], bestfit[-1]])

print(bestfit_arr)
print([[0],onesigma_all[-1][1]-bestfit[-1]])



# print(bestfit_arr[-1], bestfit_arr[9])
# from funcs import get_samples_irsic, get_bf_sigma_irsic_all
# samples_list = [samples_vid]
# legend_list = ['Fiducial new']#, "",  '${R_{s}}^{cdm}$ ($u_{0}$) 20%']#, '${R_{s}}^{cdm}$ ($u_{0}$)']#'$k$ > 0.1 s/km by 10%']#
# color_list = ['tab:orange']
# save_path = 'test_2d.pdf'
# chain_irsic = 'chain_bo_WDM_pRELICS_T0G_mru0_moC' 

# from getdist import plots
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D

# # Get Irsic+24 best fit for a chain:
# samples_irsic = get_samples_irsic(name_chain)
# bestfit, onesigma_all = get_bf_sigma_irsic_all(samples_irsic)


# print('Bestfit values', bestfit)

# # print([bestfit[-1]], [bestfit[[9]],  [ onesigma_all[-1][1]-bestfit[-1]]],[[bestfit[9] - onesigma_all[9][0]], [onesigma_all[9][1]-bestfit[9]]])
# g = plots.get_single_plotter(width_inch=16)
# g.settings.title_limit_fontsize = 15
# g.settings.axes_fontsize=15
# g.settings.axes_labelsize=15
# g.settings.legend_fontsize=15
# g.settings.fontsize = 15
# g.settings.legend_frame = False
# g.make_figure(nplot=6, nx=3, ny=3, xstretch=7, ystretch = 9)
# markersize = 11

# plt.subplots_adjust(wspace=4, top=0.95, bottom=0.05)
# # Settings
# g.settings.constrained_layout = True

# # # Set space between each subplot
# g.fig.subplots_adjust(hspace = 0.25, wspace = 0.3)

# #Reference measurements to include:
# ## reference for Boera+18 LCDM measurement                                                                            
# tau_ref=[1.02,1.41,1.69]
# dtau_high=[0.04,0.08,0.10]
# dtau_low=[0.04,0.09,0.11]
# T0_ref=[0.813,0.731,0.737]
# dT0_high=[0.134,0.135,0.167]
# dT0_low=[0.097,0.088,0.139]
# u0_ref=[7.29,7.10,4.57]
# du0_high=[0.98,0.83,1.37]
# du0_low=[1.35,1.45,1.16]

# ## reference for Rogers+21 FDM measurement                                                                            
# T0_r21_ref=[1.089,1.1330,1.0524]
# dT0_r21_high=[0.155,0.151,0.1329]
# dT0_r21_low=[0.155,0.151,0.1329]
# u0_r21_ref=[11.84,12.22,7.26]
# du0_r21_high=[5.46,4.98,2.40]
# du0_r21_low=[5.46,4.98,2.40]

# ## reference for Villasenor+22 WDM measurement                                                                        
# T0_v22_ref=[0.9073,0.8924,0.9350]
# dT0_v22_high=0.1 * np.array(T0_v22_ref) #-T0_v22_ref+[0.956042700019,0.942916828264,0.986933590842] #0.1 * T0_v22_ref    
# dT0_v22_low=0.1 * np.array(T0_v22_ref) #T0_v22_ref-[0.718568404721,0.669327643242,0.669313525854] #0.1 * T0_v22_ref      
# u0_v22_ref=[6.0878,5.603,5.2563]
# du0_v22_high=-np.array(u0_v22_ref)+np.array([6.91262758,6.41899258,6.05149578]) #0.1*u0_v22_ref                                    
# du0_v22_low=np.array(u0_v22_ref)-np.array([4.99655592,4.54443141,4.21603345]) #0.1*u0_v22_ref 
# u0_gridz42 = [4.0239696, 4.6623777, 5.6461662, 6.5922353, 7.1101564, 8.1477417, 9.9236799, 11.614156, 13.262641, 15.009738, 18.176741, 21.106591]
# u0_gridz46 = [3.6469614, 4.3163684, 5.364912, 6.3899003, 6.589134, 7.687947, 9.587889, 11.434539, 12.536159, 14.414026, 17.83328, 21.078213]
# u0_gridz5 = [2.4634363, 3.4376965, 4.3868741, 4.7135148, 5.3831631, 6.3834273, 8.1135775, 9.2448624, 9.917967, 12.360053, 15.563067, 18.783147]

# T0_value = np.arange(0.5, 1.5, 0.1)



# # z= 4.2:
# a = 10
# y1_valuez42 = 0.6 * a * T0_value**1.7
# y2_valuez42 = 1.2 * a * T0_value**1.7

# g.plot_2d(samples_list, ['T0z42','u0z42'], filled=True, ax=0,  colors = color_list)
# plt.fill_between(T0_value, y1_valuez42, y2_valuez42, color='gray', alpha=0.2)
# plt.errorbar(
# [T0_ref[0]], [u0_ref[0]],
# xerr=[[dT0_low[0]], [dT0_high[0]]], yerr=[[du0_low[0]], [du0_high[0]]],
#     fmt = '*', markersize = markersize, label='Boera+19', capsize=5, color = 'purple'
# )

# plt.errorbar(
# [T0_r21_ref[0]], [u0_r21_ref[0]],
# xerr=[[dT0_r21_low[0]], [dT0_r21_high[0]]], yerr=[[du0_r21_low[0]], [du0_r21_high[0]]],
#     fmt = '*', markersize = markersize, label='Rogers+21', capsize=5, color = 'tab:green'
# )
# plt.errorbar(
# [T0_v22_ref[0]], [u0_v22_ref[0]],
# xerr=[[dT0_v22_low[0]], [dT0_v22_high[0]]], yerr=[[du0_v22_low[0]], [du0_v22_low[0]]],
# fmt = '*', markersize = markersize,  label='Villasenor+22', capsize=5, color = 'tab:red'
# )
# # Irsic+24:
# plt.errorbar(
# [bestfit[0]], [bestfit[9]],
# xerr=[[bestfit[0] - onesigma_all[0][0]], [onesigma_all[0][1]-bestfit[0]]], yerr=[[bestfit[9] - onesigma_all[9][0]], [onesigma_all[9][1]-bestfit[9]]],
# fmt = '*', markersize = markersize,  label='Irsic+24', capsize=5, color = 'b'
# )

# # #Best-fit samples:
# # plt.scatter(bestfit_samples[0], bestfit_samples[9], marker='*', s=90, color = 'black')



# plt.ylim(np.min(u0_gridz42), np.max(u0_gridz42))
# plt.xlim(0.5, 1.4)


# # z= 4.6:
# y1_valuez46 = 0.6 * a * T0_value**1.7
# y2_valuez46 = 1.3 * a * T0_value**1.7
# g.plot_2d( samples_list, ['T0z46','u0z46'], filled=True, ax=1,  colors = color_list)
# plt.fill_between(T0_value, y1_valuez46, y2_valuez46, color='gray', alpha=0.2)
# plt.errorbar(
# [T0_ref[1]], [u0_ref[1]],
# xerr=[[dT0_low[1]], [dT0_high[1]]], yerr=[[du0_low[1]], [du0_high[1]]],
# fmt = '*', markersize = markersize, capsize=5, color = 'purple'
# )

# plt.errorbar(
# [T0_r21_ref[1]], [u0_r21_ref[1]],
# xerr=[[dT0_r21_low[1]], [dT0_r21_high[1]]], yerr=[[du0_r21_low[1]], [du0_r21_high[1]]],
# fmt = '*', markersize = markersize,  label='Rogers+21', capsize=5, color = 'tab:green'
# )
# plt.errorbar(
# [T0_v22_ref[1]], [u0_v22_ref[1]],
# xerr=[[dT0_v22_low[1]], [dT0_v22_high[1]]], yerr=[[du0_v22_low[1]], [du0_v22_low[1]]],
# fmt = '*', markersize = markersize,  label='Villasenor+22', capsize=5, color = 'tab:red'
# )


# #Irsic_24:
# plt.errorbar(
#     [bestfit[1]], [bestfit[-4]],
#     xerr=[[bestfit[1] - onesigma_all[1][0]], [ onesigma_all[1][1]-bestfit[1]]], yerr=[[bestfit[-4] - onesigma_all[-4][0]], [onesigma_all[-4][1]-bestfit[-4]]],
#     fmt = '*', markersize = markersize,  label='Irsic+24', capsize=5, color = 'b'
# )
# #     #Best-fit:
# #     plt.scatter(bestfit_samples[1], bestfit_samples[10], marker='*', s=90, color = 'black')
# #     plt.ylim(np.min(u0_gridz46), np.max(u0_gridz46))
# #     plt.xlim(0.5, 1.4)



# y1_valuez5 = 0.4 * a * T0_value**1.7
# y2_valuez5 = 1.1 * a * T0_value**1.7
# g.plot_2d(samples_list, ['T0z5','u0z5'], filled=True, ax=2,  colors = color_list)
# plt.fill_between(T0_value, y1_valuez5, y2_valuez5, color='gray', alpha=0.2)
# plt.errorbar(
# [T0_ref[2]], [u0_ref[2]],
# xerr=[[dT0_low[2]], [dT0_high[2]]], yerr=[[du0_low[2]], [du0_high[2]]],
#     fmt='*', markersize = markersize, capsize=5, color = 'purple', label = 'Boera+19')

# plt.errorbar(
# [T0_r21_ref[2]], [u0_r21_ref[2]],
# xerr=[[dT0_r21_low[2]], [dT0_r21_high[2]]], yerr=[[du0_r21_low[2]], [du0_r21_high[2]]],
# fmt = '*', markersize = markersize, label='Rogers+21', capsize=5, color = 'tab:green'
# )
# plt.errorbar(
# [T0_v22_ref[2]], [u0_v22_ref[2]],
# xerr=[[dT0_v22_low[2]], [dT0_v22_high[2]]], yerr=[[du0_v22_low[2]], [du0_v22_low[2]]],
# fmt = '*', markersize = markersize,  label= 'Villasenor+22', capsize=5, color = 'tab:red')

# plt.errorbar(
#     [bestfit[2]], [bestfit[-2]],
#     xerr=[[bestfit[2] - onesigma_all[2][0]], [ onesigma_all[2][1]-bestfit[2]]], yerr=[[bestfit[-2] - onesigma_all[-2][0]], [onesigma_all[-2][1]-bestfit[-2]]],
#     fmt = '*', markersize = markersize,  label='Irsic+24', capsize=5, color = 'b'
# )

# # #Best-fit samples:
# # plt.scatter(bestfit_samples[2], bestfit_samples[11], marker='*', s=90, color = 'black')




# custom_lines = [
# Line2D([0], [0], marker='*',color='tab:red',  label='Villasenor+22', linestyle='None',markeredgewidth=1, markersize=11),

# Line2D([0], [0], marker='*', color='tab:green',  label='Rogers+21', linestyle='None', markeredgewidth=1.5,markersize=11),

# Line2D([0], [0], marker='*', color='purple',  label='Boera+19', linestyle='None', markeredgewidth=1.5, markersize=11),
# Line2D([0], [0], marker='*', color='b',  label='Best-fit Irsic+24', linestyle='None', markeredgewidth=1.5, markersize=11),
# Line2D([0], [0], marker='*', color='black',  label='Best-fit Gaussian $T_{0}$', linestyle='None', markeredgewidth=1.5, markersize=11)
# # Line2D([0], [0], marker='v', color='black',  label='Best-fit $u_{0}$-$T_{0}$', linestyle='None', markeredgewidth=1.5, markersize=11)
# ]


# plt.legend(handles=custom_lines, loc='best', fontsize =15, frameon=False)
# plt.ylim(np.min(u0_gridz5), np.max(u0_gridz5))
# plt.xlim(0.5, 1.4)



# #Row 2:
# g.plot_2d(samples_list, ['alpha','u0z42'], filled=True, ax=3, colors = color_list) # label= {'alpha': '{m_{\mathrm{WDM}}^{-1}}'})
# g.add_legend(legend_list, legend_loc='upper right', fontsize=15, frameon=False)

# # Irsic+24 in bands:
# ax = g.fig.gca()  # Get the matplotlib axis from the getdist plotter
# plt.errorbar(
# [bestfit[-1]], [bestfit[9]],
# xerr=[[0], [ onesigma_all[-1][1]-bestfit[-1]]], yerr=[[bestfit[9] - onesigma_all[9][0]], [onesigma_all[9][1]-bestfit[9]]],
# fmt = '*', markersize = markersize,  label='Irsic+24', capsize=5, color = 'b'
# )
# # ax.axvspan(0, high_irsic['alpha'], color='darkblue', alpha=0.3)  # Add shaded region
# # ax.axvspan(0, twosig_high_irsic['alpha'], color='b', alpha=0.2)


# # #Best-fit samples:
# # plt.scatter(bestfit_samples[-2], bestfit_samples[9], marker='*', s=90, color = 'black')



# # legend_elements = [
# # mpatches.Patch(color='darkblue', alpha=0.3, label='1$\sigma$ Irsic+24'),
# # mpatches.Patch(color='b', alpha=0.3,label='2$\sigma$ Irsic+24')]
# # plt.legend(handles=legend_elements + plt.gca().get_legend_handles_labels()[0], fontsize=15, loc='upper left', bbox_to_anchor=(0.4, 1.0), frameon=False)


# g.plot_2d(samples_list, ['alpha','u0z46'], filled=True, ax=4, colors = color_list)

# # Irsic+24 in bands:
# ax = g.fig.gca()  # Get the matplotlib axis from the getdist plotter
# # ax.axvspan(0, high_irsic['alpha'], color='darkblue', alpha=0.3)  # Add shaded region
# # ax.axvspan(0, twosig_high_irsic['alpha'], color='b', alpha=0.2)


# #Best-fit samples:
# # plt.scatter(bestfit_samples[-2], bestfit_samples[10], marker='*', s=90, color = 'black')

# g.plot_2d(samples_list, ['alpha','u0z5'], filled=True, ax=5, colors = color_list)

# # Irsic+24 in bands:
# ax = g.fig.gca()  # Get the matplotlib axis from the getdist plotter
# # ax.axvspan(0, high_irsic['alpha'], color='darkblue', alpha=0.3)  # Add shaded region
# # ax.axvspan(0, twosig_high_irsic['alpha'], color='b', alpha=0.2)


# # #Best-fit samples:
# # plt.scatter(bestfit_samples[-2], bestfit_samples[11], marker='*', s=90, color = 'black')

# #Row 3:
# g.plot_2d(samples_list, ['fwdm','u0z42'], filled=True, ax=6, colors = color_list)

# #Best-fit samples:
# # plt.scatter(bestfit_samples[-1], bestfit_samples[9], marker='*', s=90, color = 'black')


# g.plot_2d(samples_list, ['fwdm','u0z46'], filled=True, ax=7, colors = color_list)

# # #Best-fit samples:
# # plt.scatter(bestfit_samples[-1], bestfit_samples[10], marker='*', s=90, color = 'black')

# g.plot_2d(samples_list, ['fwdm','u0z5'], filled=True, ax=8, colors = color_list)

# #Best-fit samples:
# # plt.scatter(bestfit_samples[-1], bestfit_samples[11], marker='*', s=90, color = 'black')

# if save_path != None:
#     plt.savefig(save_path, bbox_inches='tight') 


# %%
