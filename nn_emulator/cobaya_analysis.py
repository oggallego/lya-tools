import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt, ticker as mticker
import numpy as np
from getdist import MCSamples
import pandas as pd
from getdist import parampriors

from getdist import MCSamples
from getdist import plots

import sys
sys.path.append('/data/og313/lyalpha/code_wdm/nn_emulator_fwdm')
from model_loader import get_mixed_models
from predictor import get_nn_prediction_mcmc
import os

def get_info_for_cornerplot():
    param_bounds = parampriors.ParamBounds()

    taueff_boeraz42 = 1.56 * ((1 + 4.2) / 5.75)** 4
    taueff_boeraz46 = 1.56 * ((1 + 4.6) / 5.75)** 4
    taueff_boeraz5 = 1.56 * ((1 + 5) / 5.75)** 4

    # Manually set the names, lower, and upper attributes
    param_bounds.names =     ['T0z42', 'T0z46', 'T0z5',
    'gammaz42', 'gammaz46', 'gammaz5',
    'taueffz42', 'taueffz46', 'taueffz5',
    'u0z42', 'u0z46', 'u0z5',
    'alpha', 'fwdm']
                
    param_bounds.lower = {'T0z42': 0.5, 'gammaz42': 0.9, 'taueffz42': 0.3*taueff_boeraz42, 'u0z42': 4.0239696, 
                          'T0z46': 0.5, 'gammaz46': 0.9, 'taueffz46': 0.3*taueff_boeraz46, 'u0z46': 3.6469614, 
                          'T0z5': 0.5, 'gammaz5': 0.9, 'taueffz5': 0.3*taueff_boeraz5, 'u0z5': 2.4634363,
                          'alpha': 0, 'fwdm': 0.0}
    param_bounds.upper = {'T0z42': 1.4, 'gammaz42': 1.8, 'taueffz42': 1.7*taueff_boeraz42, 'u0z42':21.106591,
                          'T0z46': 1.4, 'gammaz46': 1.8, 'taueffz46': 1.7*taueff_boeraz46, 'u0z46': 21.078213,
                          'T0z5': 1.4, 'gammaz5': 1.8, 'taueffz5': 1.7*taueff_boeraz5, 'u0z5': 18.783147,
                         'alpha': 1e30, 'fwdm': 1.0}
    # print(param_bounds.names)
    # Now you can create the ranges dictionary
    ranges = {name: (param_bounds.lower[name], param_bounds.upper[name]) for name in param_bounds.names}

    # Define the labels:
    labels = {
    'T0z42': '{T_{0}}^{\,4.2}/{10^{4}} \,\\mathrm{(K)}',
    'T0z46': '{T_{0}}^{\,4.6}/{10^{4}} \,\\mathrm{(K)}',
    'T0z5': '{T_{0}}^{\,5}/{10^{4}} \,\\mathrm{(K)}',
    'gammaz42': '{\\gamma}^{\,4.2}',
    'gammaz46': '{\\gamma}^{\,4.6}',
    'gammaz5': '{\\gamma}^{\,5}',
    'taueffz42': '{\\tau_{\\mathrm{eff}}}^{\,4.2}',
    'taueffz46': '{\\tau_{\\mathrm{eff}}}^{\,4.6}',
    'taueffz5': '{\\tau_{\\mathrm{eff}}}^{\,5}',
    'u0z42': 'u_{0}^{\,4.2}\,\\mathrm{(eV\,m_{p}^{-1})}',
    'u0z46': 'u_{0}^{\,4.6}\,\\mathrm{(eV\,m_{p}^{-1})}',
    'u0z5': 'u_{0}^{\,5}\,\\mathrm{(eV\,m_{p}^{-1})}',
    'alpha': '{m_{\\mathrm{WDM}}}^{-1}\,\\mathrm{(keV^{-1})}',
    'fwdm': 'f_{\\mathrm{WDM}}'
    }

    return ranges, list(ranges.keys()), list(labels.values())

def get_samples_from_pd(full_path, burning, ign_rows, alpha_max):

    ranges, names, labels = get_info_for_cornerplot()
    # print('Labels:', labels)
    'No burn-in added here'

    with open(f'{full_path}', 'r') as file:
        first_line = file.readline().strip()

    # Split the first line by whitespace to get the column names
    column_names = first_line.split()


    param_columns = [
        'T0z42', 'T0z46', 'T0z5',
        'gammaz42', 'gammaz46', 'gammaz5',
        'taueffz42', 'taueffz46', 'taueffz5',
        'u0z42', 'u0z46', 'u0z5',
        'alpha', 'fwdm'
    ]

    # Read the data from the file
    df = pd.read_csv(f'{full_path}', delim_whitespace=True, header=None, skiprows=1,engine='python')
    # print(f"DataFrame shape: {df.shape}")
    df.columns = column_names[1:]


    # Extract relevant columns
    chain_values = df[param_columns].values
    weights = df['weight'].values
    loglike = -df['minuslogpost'].values

    if burning == True:
        mc_samples = MCSamples(samples=chain_values, names=param_columns, labels=labels, weights = weights, loglikes = loglike, ranges = ranges, ignore_rows = ign_rows)
    else: 
        mc_samples = MCSamples(samples=chain_values, names=param_columns, labels=labels, weights = weights, loglikes = loglike, ranges = ranges)

    # Set ranges for samples:
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
    'alpha': (0, alpha_max),
    'fwdm': (0, 1)}
    # Set the ranges for the parameters
    mc_samples.setRanges(param_ranges)
    
    return mc_samples

def compute_chi_squared(residuals, cov_matrix):
    # Compute the inverse of the covariance matrix
    cov_matrix_inv = np.linalg.inv(cov_matrix)
    
    # Compute the chi-squared value
    chi2 = np.dot(residuals.T, np.dot(cov_matrix_inv, residuals))
    
    return chi2


def compute_log_likelihood(residuals, cov_matrix):
    # Compute the chi-squared value
    chi2 = compute_chi_squared(residuals, cov_matrix)
    
    # Compute the log-likelihood
    log_likelihood_value = -0.5 * chi2
    log_likelihood_value -= 0.5 * np.log(np.linalg.det(cov_matrix))
    log_likelihood_value -= (np.shape(cov_matrix)[0]) * 0.5 * np.log(2 * np.pi)
    
    return log_likelihood_value

import os

def prepare_bestfit_and_models(samples, redshifts, n_pars=14):
    """
    Prepare best-fit samples and load emulators for multiple redshifts.

    Args:
        samples (object): Samples object containing the likelihood statistics.
        redshifts (list): List of redshifts for which models need to be prepared.
        n_pars (int, optional): Number of parameters for best-fit samples. Default is 14.

    Returns:
        dict: A dictionary containing:
            - "bestfit_samples": List of best-fit parameter values.
            - "models": Dictionary of models for each redshift.
            - "min_in": Dictionary of minimum inputs for each redshift.
            - "max_in": Dictionary of maximum inputs for each redshift.
            - "taueff_boera": Dictionary of tau_eff values for each redshift.
    """
    # Get best-fit samples
    like_stats = samples.getLikeStats()
    bestfit_samples = [like_stats.names[i].bestfit_sample for i in range(n_pars)]
    print('Best-fit samples:', bestfit_samples)

    # Prepare models and associated data for each redshift
    current_directory = os.getcwd()
    print('Current directory is', current_directory)
    models = {}
    min_in = {}
    max_in = {}
    taueff_boera = {}

    for z in redshifts:
        # Dynamically generate input/output file names
        in_file = f"inputs{z}{current_directory.split('/')[-1].split('analysis')[1]}.txt"
        out_file = f"ps{z}{current_directory.split('/')[-1].split('analysis')[1]}.txt"
        print(f"Input file for z = {z}: {in_file}")
        print(f"Output file for z = {z}: {out_file}")

        # Load models and associated data
        models[z], min_in[z], max_in[z], taueff_boera[z] = get_mixed_models(z, current_directory, in_file, out_file)

    # Return all prepared data
    return {
        "bestfit_samples": bestfit_samples,
        "models": models,
        "min_in": min_in,
        "max_in": max_in,
        "taueff_boera": taueff_boera
    }

def compute_chi_squared_and_covariance(prepared_data, redshifts, P_kRC, covariance_matrix):
    """
    Compute chi-squared, log-likelihood, and covariance matrices for multiple redshifts.

    Args:
        prepared_data (dict): Output of `prepare_bestfit_and_models` containing best-fit samples and models.
        redshifts (list): List of redshifts for which computations are performed.
        P_kRC (dict): Power spectrum data for each redshift.
        covariance_matrix (numpy.ndarray): Full covariance matrix.

    Returns:
        dict: Dictionary containing residuals, covariance matrices, chi-squared values,
              log-likelihood values, and predictions for each redshift.
    """
    bestfit_samples = prepared_data["bestfit_samples"]
    models = prepared_data["models"]
    min_in = prepared_data["min_in"]
    max_in = prepared_data["max_in"]
    taueff_boera = prepared_data["taueff_boera"]

    results = {}

    for i, z in enumerate(redshifts):
        # Compute best-fit input for this redshift
        bf_input = [
            bestfit_samples[0 + i],
            bestfit_samples[3 + i],
            bestfit_samples[6 + i],
            bestfit_samples[9 + i],
            bestfit_samples[12],
            bestfit_samples[13]
        ]

        print(f"Best-fit values in order at z = {z}: {bf_input}")

        # Generate predictions
        bf_pred = get_nn_prediction_mcmc(
            z, bf_input, min_in[z], max_in[z], models[z], rc=False) 

        # Compute residuals and extract the appropriate covariance matrix
        residuals = bf_pred - P_kRC[z]
        residuals = residuals.reshape(-1, 1)
        cov_indices = slice(i * 16, (i + 1) * 16)
        cov_matrix = covariance_matrix[cov_indices, cov_indices]

        # Compute chi-squared and log-likelihood
        chi_value = compute_chi_squared(residuals, cov_matrix)
        log_likelihood_value = compute_log_likelihood(residuals, cov_matrix)

        print(f"Chi-squared for z = {z}: {chi_value}")
        print(f"Log-likelihood for z = {z}: {log_likelihood_value}")

        # Store results for this redshift
        results[z] = {
            "bf_pred": bf_pred,
            "residuals": residuals,
            "cov_matrix": cov_matrix,
            "chi_value": chi_value,
            "log_likelihood": log_likelihood_value,
            "P_kRC": P_kRC[z]
        }

        print(f'Before plot_power_spectra, covariance matrix for z = {z}: {np.sqrt(np.diag(results[z]["cov_matrix"]))}')

    return results


def plot_power_spectra(log_k, k, data, output_file=None):
    """
    Plots power spectra and residuals for multiple redshifts.

    Args:
        log_k (numpy.ndarray): Logarithmic k values.
        k (numpy.ndarray): k values (linear scale).
        data (dict): Dictionary containing data for each redshift.
                     Example:
                     {
                         4.2: {
                             "P_kRC": np.ndarray,
                             "bf_pred": np.ndarray,
                             "chi_value": float,
                             "color": str,
                             "cov_matrix(z)": np.ndarray
                         },
                         ...
                     }
        covariance_matrix (numpy.ndarray): Full covariance matrix.
        output_file (str, optional): File path to save the plot. If None, the plot is displayed.
    """
    fig, ax = plt.subplots(2, 3, sharex=True, figsize=(40, 20))
    plt.rcParams.update({'font.size': 30})

    for i, (z, z_data) in enumerate(data.items()):
        row, col = divmod(i, 3)

        # Extract data for this redshift
        P_kRC = z_data["P_kRC"]
        bf_pred = z_data["bf_pred"]
        chi_value = z_data["chi_value"]
        color = z_data["color"]
        cov_matrix = z_data["covariance_matrix"]
        print(f"Covariance matrix for z = {z}: {np.sqrt(np.diag(cov_matrix))}")


        if isinstance(chi_value, np.ndarray):
            chi_value = chi_value.item()
        print(f'Before plotting, covariance matrix for z = {z}: {np.sqrt(np.diag(cov_matrix))}')

        # Flatten the arrays if necessary
        P_kRC = P_kRC.flatten()

        bf_pred = bf_pred.flatten()

        # Top plot: Power spectrum
        ax[0][col].plot(log_k, bf_pred * k / np.pi, linestyle='--',
                        label=f'z = {z}, $\\chi^{{2}} = {round(chi_value, 2)}$', color=color)
        ax[0][col].errorbar(log_k, P_kRC * k / np.pi,
                            yerr=(k / np.pi) * np.sqrt(cov_matrix.diagonal()),
                            fmt='D', markersize=4, color=color)
        ax[0][col].set_ylabel('kP(k)$/\\pi$')
        ax[0][col].set_xlabel('$\\mathrm{log}_{10}$ ($k$ $\\mathrm{[km^{-1}\\,s]})$')
        ax[0][col].set_ylim(0.01, 0.8)
        ax[0][col].set_xlim(-2.25, -0.65)
        ax[0][col].set_yscale('log')
        ax[0][col].xaxis.set_major_locator(mticker.MultipleLocator(0.5))
        ax[0][col].xaxis.set_minor_locator(mticker.MultipleLocator(0.1))
        ax[0][col].legend()

        # Bottom plot: Residuals
        ax[1][col].errorbar(log_k, P_kRC / bf_pred,
                            yerr=np.sqrt(cov_matrix.diagonal()) / bf_pred,
                            fmt='D', markersize=4, color=color)
        ax[1][col].axhline(y=1, linestyle='--', color='tab:grey')
        ax[1][col].set_ylabel('Residuals')
        ax[1][col].set_xlabel('$\\mathrm{log}_{10}$ ($k$ $\\mathrm{[km^{-1}\\,s]})$')
        ax[1][col].legend()

    plt.subplots_adjust(right=0.95, wspace=0.25)
    if output_file:
        plt.savefig(output_file)
        print('Saved figure in', output_file)
    else:
        plt.show()


def get_alpha_vs_fwdm_2d(samples_list, color_list, legend_list, save_path):
    g = plots.get_subplot_plotter(subplot_size=2, width_inch=6)
    print('Changes saved')
    g.settings.axes_fontsize=15
    g.settings.axes_labelsize=17
    g.settings.legend_fontsize=15
    g.settings.fontsize = 15
    g.plot_2d(
        samples_list, 'alpha',
        'fwdm', filled=True, colors = color_list)
    
    g.add_legend(legend_list, legend_loc='upper right', fontsize = 15)
    g.export(f'{save_path}')
        
def u0_vs__T0_alpha_fwdm(samples_list, chain_irsic, color_list, legend_list, save_path):

    # 2D plot:
    from getdist import plots
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    g = plots.get_single_plotter(width_inch=16)
    g.settings.title_limit_fontsize = 15
    g.settings.axes_fontsize=15
    g.settings.axes_labelsize=15
    g.settings.legend_fontsize=15
    g.settings.fontsize = 15
    g.settings.legend_frame = False
    g.make_figure(nplot=6, nx=3, ny=3, xstretch=7, ystretch = 9)
    markersize = 11

    plt.subplots_adjust(wspace=4, top=0.95, bottom=0.05)
    # Settings
    g.settings.constrained_layout = True

    # # Set space between each subplot
    g.fig.subplots_adjust(hspace = 0.25, wspace = 0.3)

    #Reference measurements to include:
    ## reference for Boera+18 LCDM measurement                                                                            
    tau_ref=[1.02,1.41,1.69]
    dtau_high=[0.04,0.08,0.10]
    dtau_low=[0.04,0.09,0.11]
    T0_ref=[0.813,0.731,0.737]
    dT0_high=[0.134,0.135,0.167]
    dT0_low=[0.097,0.088,0.139]
    u0_ref=[7.29,7.10,4.57]
    du0_high=[0.98,0.83,1.37]
    du0_low=[1.35,1.45,1.16]

    ## reference for Rogers+21 FDM measurement                                                                            
    T0_r21_ref=[1.089,1.1330,1.0524]
    dT0_r21_high=[0.155,0.151,0.1329]
    dT0_r21_low=[0.155,0.151,0.1329]
    u0_r21_ref=[11.84,12.22,7.26]
    du0_r21_high=[5.46,4.98,2.40]
    du0_r21_low=[5.46,4.98,2.40]

    ## reference for Villasenor+22 WDM measurement                                                                        
    T0_v22_ref=[0.9073,0.8924,0.9350]
    dT0_v22_high=0.1 * np.array(T0_v22_ref) #-T0_v22_ref+[0.956042700019,0.942916828264,0.986933590842] #0.1 * T0_v22_ref    
    dT0_v22_low=0.1 * np.array(T0_v22_ref) #T0_v22_ref-[0.718568404721,0.669327643242,0.669313525854] #0.1 * T0_v22_ref      
    u0_v22_ref=[6.0878,5.603,5.2563]
    du0_v22_high=-np.array(u0_v22_ref)+np.array([6.91262758,6.41899258,6.05149578]) #0.1*u0_v22_ref                                    
    du0_v22_low=np.array(u0_v22_ref)-np.array([4.99655592,4.54443141,4.21603345]) #0.1*u0_v22_ref 
    u0_gridz42 = [4.0239696, 4.6623777, 5.6461662, 6.5922353, 7.1101564, 8.1477417, 9.9236799, 11.614156, 13.262641, 15.009738, 18.176741, 21.106591]
    u0_gridz46 = [3.6469614, 4.3163684, 5.364912, 6.3899003, 6.589134, 7.687947, 9.587889, 11.434539, 12.536159, 14.414026, 17.83328, 21.078213]
    u0_gridz5 = [2.4634363, 3.4376965, 4.3868741, 4.7135148, 5.3831631, 6.3834273, 8.1135775, 9.2448624, 9.917967, 12.360053, 15.563067, 18.783147]
    
    ##Irsic+24 obtained from u0T0_mru0_C.txt chains
    # bf_irsic, low_irsic, high_irsic = get_irsic_bf(chain_irsic)
    # # print(f'Best fit values from Irsic, {bf_irsic}')
    # print('Errors going into alpha plot', print([bf_irsic['u0z42']], [bf_irsic['alpha']]))

    # low_irsic = np.abs(low_irsic)
    # high_irsic = np.abs(high_irsic)
    # print(low_irsic)

    low = [0.10190820067131767, 0.19355134962665776, 0.06713370699868626, 0.12922874221253933, -0.038459409115058296, 0.1436790794500722, 0.017235194825470757, 0.012378019732014689, 0.14545815300408393, 0.6884616607048581, -0.33068792255343027, 0, 0]
    high = [0.10179350087130234, 0.01568884043485963, 0.1296260393030222, 0.21298329448391495, 0.43902050778013413, 0.2502656146928892, 0.10815572912229987, 0.11673382469830429, 0.037605095914419184, 1.7721204294570922, 2.288923007360297, 0.9027575377272306, -0.11628441327192107]
    #Row 1:
    T0_value = np.arange(0.5, 1.5, 0.1)

    # print('Errors going into alpha plot', print([bf_irsic[-2]], [bf_irsic[9]]))

    # z= 4.2:
    a = 10
    y1_valuez42 = 0.6 * a * T0_value**1.7
    y2_valuez42 = 1.2 * a * T0_value**1.7
    g.plot_2d(samples_list, ['T0z42','u0z42'], filled=True, ax=0,  colors = color_list)
    plt.fill_between(T0_value, y1_valuez42, y2_valuez42, color='gray', alpha=0.2)
    plt.errorbar(
        [T0_ref[0]], [u0_ref[0]],
        xerr=[[dT0_low[0]], [dT0_high[0]]], yerr=[[du0_low[0]], [du0_high[0]]],
         fmt = '*', markersize = markersize, label='Boera+19', capsize=5, color = 'purple'
    )

    # plt.errorbar(
    #     [bf_irsic[0]], [np.abs(bf_irsic[9])],
    #     xerr=[[low_irsic[0]], [high_irsic[0]]], yerr=[[np.abs(low_irsic[9])], [np.abs(high_irsic[9])]],
    #      fmt = '*', markersize = markersize, label='Irsic+24', capsize=5, color = 'chocolate'
    # )

    plt.errorbar(
        [T0_r21_ref[0]], [u0_r21_ref[0]],
        xerr=[[dT0_r21_low[0]], [dT0_r21_high[0]]], yerr=[[du0_r21_low[0]], [du0_r21_high[0]]],
         fmt = '*', markersize = markersize, label='Rogers+21', capsize=5, color = 'tab:green'
    )
    plt.errorbar(
        [T0_v22_ref[0]], [u0_v22_ref[0]],
        xerr=[[dT0_v22_low[0]], [dT0_v22_high[0]]], yerr=[[du0_v22_low[0]], [du0_v22_low[0]]],
        fmt = '*', markersize = markersize,  label='Villasenor+22', capsize=5, color = 'tab:red'
    )

    plt.ylim(np.min(u0_gridz42), np.max(u0_gridz42))
    plt.xlim(0.5, 1.4)


    # z= 4.6:
    y1_valuez46 = 0.6 * a * T0_value**1.7
    y2_valuez46 = 1.3 * a * T0_value**1.7
    g.plot_2d( samples_list, ['T0z46','u0z46'], filled=True, ax=1,  colors = color_list)
    plt.fill_between(T0_value, y1_valuez46, y2_valuez46, color='gray', alpha=0.2)
    plt.errorbar(
        [T0_ref[1]], [u0_ref[1]],
        xerr=[[dT0_low[1]], [dT0_high[1]]], yerr=[[du0_low[1]], [du0_high[1]]],
        fmt = '*', markersize = markersize, capsize=5, color = 'purple'
    )

    # plt.errorbar(
    #     [bf_irsic['T0z46']], [np.abs(bf_irsic['u0z46'])],
    #     xerr=[[low_irsic['T0z46']], [high_irsic['T0z46']]], yerr=[np.abs([low_irsic['u0z46']]), [np.abs(high_irsic['u0z46'])]],
    #     fmt = '*', markersize = markersize,  label='Irsic+24', capsize=5, color = 'chocolate'
    # )
    plt.errorbar(
        [T0_r21_ref[1]], [u0_r21_ref[1]],
        xerr=[[dT0_r21_low[1]], [dT0_r21_high[1]]], yerr=[[du0_r21_low[1]], [du0_r21_high[1]]],
        fmt = '*', markersize = markersize,  label='Rogers+21', capsize=5, color = 'tab:green'
    )
    plt.errorbar(
        [T0_v22_ref[1]], [u0_v22_ref[1]],
        xerr=[[dT0_v22_low[1]], [dT0_v22_high[1]]], yerr=[[du0_v22_low[1]], [du0_v22_low[1]]],
        fmt = '*', markersize = markersize,  label='Villasenor+22', capsize=5, color = 'tab:red'
    )
    plt.ylim(np.min(u0_gridz46), np.max(u0_gridz46))
    plt.xlim(0.5, 1.4)

    y1_valuez5 = 0.4 * a * T0_value**1.7
    y2_valuez5 = 1.1 * a * T0_value**1.7
    g.plot_2d(samples_list, ['T0z5','u0z5'], filled=True, ax=2,  colors = color_list)
    plt.fill_between(T0_value, y1_valuez5, y2_valuez5, color='gray', alpha=0.2)
    plt.errorbar(
        [T0_ref[2]], [u0_ref[2]],
        xerr=[[dT0_low[2]], [dT0_high[2]]], yerr=[[du0_low[2]], [du0_high[2]]],
            fmt='*', markersize = markersize, capsize=5, color = 'purple', label = 'Boera+19')

    
    # plt.errorbar(
    #     [bf_irsic['T0z5']], [np.abs(bf_irsic['u0z5'])],
    #     xerr=[[low_irsic['T0z5']], [high_irsic['T0z5']]], yerr=[np.abs([low_irsic['u0z5']]), [np.abs(high_irsic['u0z5'])]],
    #     fmt = '*', markersize = markersize,  label='Irsic+24', capsize=5, color = 'chocolate'
    # )
    plt.errorbar(
        [T0_r21_ref[2]], [u0_r21_ref[2]],
        xerr=[[dT0_r21_low[2]], [dT0_r21_high[2]]], yerr=[[du0_r21_low[2]], [du0_r21_high[2]]],
        fmt = '*', markersize = markersize, label='Rogers+21', capsize=5, color = 'tab:green'
    )
    plt.errorbar(
        [T0_v22_ref[2]], [u0_v22_ref[2]],
        xerr=[[dT0_v22_low[2]], [dT0_v22_high[2]]], yerr=[[du0_v22_low[2]], [du0_v22_low[2]]],
        fmt = '*', markersize = markersize,  label= 'Villasenor+22', capsize=5, color = 'tab:red')



    custom_lines = [
        Line2D([0], [0], marker='*',color='tab:red',  label='Villasenor+22', linestyle='None',markeredgewidth=1, markersize=11),
        
        Line2D([0], [0], marker='*', color='tab:green',  label='Rogers+21', linestyle='None', markeredgewidth=1.5,markersize=11),
        
        Line2D([0], [0], marker='*', color='purple',  label='Boera+19', linestyle='None', markeredgewidth=1.5, markersize=11)
       # Line2D([0], [0], marker='*', color='chocolate',  label='Irsic+24', linestyle='None', markeredgewidth=1.5, markersize=11),
    ]

    # custom_caps = Line2D([0], [0], color='blue', linestyle='None', marker='_', 
    #                   markersize=8, label='Error Caps')

    # custom_combined = Line2D(
    #     [0], [0], marker='o', color='tab:blue', markersize=6, linestyle='None',
    #     markerfacecolor='tab:blue', markeredgecolor='tab:blue', label='Data with Error Bars'
    # )

    # Add a line to represent error bars
    


    plt.legend(handles=custom_lines, loc='best', fontsize =15, frameon=False)
    plt.ylim(np.min(u0_gridz5), np.max(u0_gridz5))
    plt.xlim(0.5, 1.4)



    #Row 2:
    g.plot_2d(samples_list, ['alpha','u0z42'], filled=True, ax=3, colors = color_list) # label= {'alpha': '{m_{\mathrm{WDM}}^{-1}}'})
    g.add_legend(legend_list, legend_loc='upper right', fontsize=15, frameon=False)
    # plt.errorbar(
    #     [np.abs(bf_irsic['alpha'])], [np.abs(bf_irsic['u0z42'])],
    #     xerr=[[0],[np.abs(high_irsic['alpha'])]], yerr=[np.abs([low_irsic['u0z42']]), [np.abs(high_irsic['u0z42'])]],
    #     fmt = '*', markersize = markersize,  label='Irsic+24', capsize=5, color = 'chocolate'
    # )
    # print([bf_irsic[-2]], [bf_irsic[9]])
    # custom_lines = [
    #     Line2D([0], [0], marker='o', color='chocolate',  label='Irsic+24', linestyle='None', markeredgewidth=1.5)
    # ]
    # plt.legend(handles=custom_lines, loc='best')

    g.plot_2d(samples_list, ['alpha','u0z46'], filled=True, ax=4, colors = color_list)
    # plt.errorbar(
    #     [np.abs(bf_irsic['alpha'])], [np.abs(bf_irsic['u0z46'])],
    #     xerr=[[0],[np.abs(high_irsic['alpha'])]], yerr=[np.abs([low_irsic['u0z46']]), [np.abs(high_irsic['u0z46'])]],
    #     fmt = '*', markersize = markersize,  label='Irsic+24', capsize=5, color = 'chocolate'
    # )

    g.plot_2d(samples_list, ['alpha','u0z5'], filled=True, ax=5, colors = color_list)
    # plt.errorbar(
    #     [np.abs(bf_irsic['alpha'])], [np.abs(bf_irsic['u0z5'])],
    #     xerr=[[0],[np.abs(high_irsic['alpha'])]], yerr=[np.abs([low_irsic['u0z5']]), [np.abs(high_irsic['u0z5'])]],
    #     fmt = '*', markersize = markersize,  label='Irsic+24', capsize=5, color = 'chocolate'
    # )

    # custom_lines = [
    # Line2D([0], [0], marker='*', color='chocolate',  label='Irsic+24', linestyle='None',markeredgewidth=1.5, markersize=11)
    # ]
    
    # plt.legend(handles=custom_lines, loc='best', fontsize =15, frameon=False)
   
    #Row 3:
    g.plot_2d(samples_list, ['fwdm','u0z42'], filled=True, ax=6, colors = color_list)
    g.plot_2d(samples_list, ['fwdm','u0z46'], filled=True, ax=7, colors = color_list)
    g.plot_2d(samples_list, ['fwdm','u0z5'], filled=True, ax=8, colors = color_list)

    if save_path != None:
        plt.savefig(f'{save_path}')
