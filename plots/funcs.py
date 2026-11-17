import sys
sys.path.append('/data/og313/lyalpha/code_wdm/nn_emulator_fwdm')
from model import NNInterpolator
from predictor import get_nn_prediction_mcmc, get_nn_prediction_mcmc_noise, get_nn_prediction_post_rs_mcmc
from cobaya_analysis import compute_chi_squared
sys.path.append('/data/og313/lyalpha/code_wdm/mcmc')
from get_boera_data import get_data
from getdist import loadMCSamples
import numpy as np
import os
import torch
import pandas as pd
from getdist import MCSamples
import matplotlib.patches as mpatches

import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt, ticker as mticker
import matplotlib.ticker as ticker

def get_samples_smooth(directory, burning):
    """
    Get samples from array job - chain name format format from _{i}.1.txt and dirrctory
    Directory needs to contain chain.paramnames and chain.ranges
    """
    smooth_2D = 0.7
    smooth_1D = 0.7
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

def get_samples_irsic(name_chain):
    """
    Get samples from Vid's chains
    """
    #directory = f'/data/vault/og313/chains_irsic/{name_chain}'
    directory = name_chain
    smooth_2D = 0.5
    smooth_1D = 0.5
    ignore_rows = 0.8

    # Orden of names in Vid's saved chains


    # Load the samples without altering parameter names yet
    # To run this need .chainparamnames in the same folder
    samples_vid = loadMCSamples(directory, settings={
        'ignore_rows': ignore_rows,
        'smooth_scale_2D': smooth_2D,
        'smooth_scale_1D': smooth_1D
    })

    # # Set the original parameter names
    # samples_vid.setParamNames(original_names)


    print('Gelman-Rubin ratio', samples_vid.getGelmanRubin())
    total_samples = samples_vid.samples.shape[0] #ids 1 corresponds to number of parameters
    print('Total number of samples is', total_samples)

    return samples_vid

def get_sigma_irsic_alpha(samples_irsic):
    sigma = 1-0.682689492137086

    ### more accurately
    two_sigma = 1-0.954499736103642
    three_sigma = 1-0.997300203936740
    four_sigma = 1-0.999936657516334# = 4sigma
    five_sigma = 1-0.999999426696856 #= 5sigma

    upper_bound_sig = (samples_irsic.confidence('alpha', limfrac=sigma, upper=True))
    upper_bound_twosig = (samples_irsic.confidence('alpha', limfrac=two_sigma, upper=True))
    print('mwdm <', 1/upper_bound_twosig, 'keV  is excluded at 2 sigman on Irsic+24')

    return upper_bound_sig, upper_bound_twosig


def get_bf_sigma_irsic_all(name_chain, samples_irsic):
    stats = samples_irsic.getMargeStats()

    names =  [
    'T0z42', 'T0z46', 'T0z5',
    'gammaz42', 'gammaz46', 'gammaz5',
    'taueffz42', 'taueffz46', 'taueffz5',
    'u0z42', 'u0z46', 'u0z5',
    'alpha' ]
    if 'nofqso' in name_chain:
        original_names = ['taueffz42','taueffz46', 'taueffz5', 'T0z42', 'T0z46', 'T0z5',
                    'gammaz42', 'gammaz46', 'gammaz5', 
                    'u0z42', 'u0z46', 'u0z5',
                    'sigma8', 'neff', 'alpha']# 'fqso']
    if 'An' in name_chain:
        original_names = ['taueffz42','taueffz46', 'taueffz5', 'T0z42', 'T0z46', 'T0z5',
                    'gammaz42', 'gammaz46', 'gammaz5',
                    'u0z42', 'u0z46', 'u0z5',
                          'sigma8', 'neff', 'alpha', 'fqso', 'Anz42', 'Anz46', 'Anz5']
    else: 
        original_names = ['taueffz42','taueffz46', 'taueffz5', 'T0z42', 'T0z46', 'T0z5',
                    'gammaz42', 'gammaz46', 'gammaz5', 
                    'u0z42', 'u0z46', 'u0z5',
                    'sigma8', 'neff', 'alpha', 'fqso']

    names_ordered = ['T0z42' 'T0z46', 'T0z5', 'gammaz42', 'gammaz46', 'gammaz5', 'taueffz42', 'taueffz46', 'taueffz5', 'u0z42', 'u0z46', 'u0z5']
    onesigma_all = []
    for i in original_names:
        print(i)
        lims = stats.parWithName(f'{i}').limits
        for conf, lims in zip(samples_irsic.contours,lims):
            # (conf, lim0.lower, lim0.upper,
            # print(f'Parameter {i}: {conf}, {lims.lower} and {lims.upper}')
            if conf == 0.68:
                onesigma_all.append((lims.lower, lims.upper))
            # [0] T0z42, [1] T0 z46 ...
    print('One sigma arra', onesigma_all)
    if 'nofqso' in name_chain:
        onesigma_all = np.array([onesigma_all[3], onesigma_all[4], onesigma_all[5], onesigma_all[6],
                                 onesigma_all[7], onesigma_all[8], onesigma_all[0], onesigma_all[1],
                                 onesigma_all[2], onesigma_all[9], onesigma_all[10], onesigma_all[11],
                                onesigma_all[-1]])
    else:
        onesigma_all = np.array([onesigma_all[3], onesigma_all[4], onesigma_all[5], onesigma_all[6],
                                 onesigma_all[7], onesigma_all[8], onesigma_all[0], onesigma_all[1],
                                 onesigma_all[2], onesigma_all[9], onesigma_all[10], onesigma_all[11],
                                onesigma_all[-2]])
    print('Sigma of T0z46', onesigma_all[2][0], onesigma_all[2][1])
    bestfit = np.zeros(len(original_names))
    result = samples_irsic.getLikeStats()
    for i in range(len(bestfit)):
        bestfit[i] = result.names[i].bestfit_sample
    print('Best fit total array before ordering', bestfit)
    if 'nofqso' in name_chain:
        bestfit_arr = np.array([bestfit[3], bestfit[4], bestfit[5], bestfit[6], bestfit[7],
    bestfit[8], bestfit[0], bestfit[1], bestfit[2], bestfit[9], bestfit[10],
    bestfit[11], bestfit[-1]])
    if 'An' in name_chain:
        bestfit_arr = np.array([bestfit[3], bestfit[4], bestfit[5], bestfit[6], bestfit[7],
    bestfit[8], bestfit[0], bestfit[1], bestfit[2], bestfit[9], bestfit[10],
                                bestfit[11], bestfit[-5],bestfit[-3], bestfit[-2], bestfit[-1]])
    else:
        bestfit_arr = np.array([bestfit[3], bestfit[4], bestfit[5], bestfit[6], bestfit[7],
    bestfit[8], bestfit[0], bestfit[1], bestfit[2], bestfit[9], bestfit[10],
    bestfit[11], bestfit[-2]])
    print('Best fit from Irsic', bestfit_arr)

  #  bestfit_arr = [0.820007, 1.01994, 1.09818, 6.54541,
  #             0.76636, 1.53897, 1.52697,6.00114, 
  #          0.946699, 1.08196, 1.60832, 3.12389,
  #             0.0439271]
    

    return bestfit_arr, np.array(onesigma_all)


def get_models_z(z, base_path):
    """
    Function to load mixed models different from predictor.py 
    Need preprocessing to know min max normalization to apply to input 
    """
    
    # inputs = np.loadtxt(f'{base_path}/z{z}/inputs{z}{base_path.split("analysis")[1]}_all.txt')
    # print('Shape of all inputs loaded', np.shape(inputs))

    # labels = np.loadtxt(f'{base_path}/z{z}/ps{z}{base_path.split("analysis")[1]}_all.txt')
    # print('Shape of all inputs loaded', np.shape(labels))

    learning_rate = 0.0001  # Fiducial value
    patience = 30
    hidden = 120
    act = torch.nn.ReLU
    model = NNInterpolator(
    input_size=6,
    output_size=16,
    hidden=hidden,
    learning_rate=learning_rate,
    act=act,
    patience=patience,
    patience_scheduler=30,  # Not needed for inference
    base_path=base_path)

    models = []

    # _, _, min_in, max_in = model.preprocess_data(z, inputs, np.log(labels), batch_size=18)
    # print('Made it here')

    norm = np.loadtxt(f'{base_path}/norm_{z}.txt')
    min_in = norm[0]
    max_in = norm[1]

    #     # Load models from saved states
    base_path_emulator = os.path.join(base_path, f"z{z}")
    print('Base path of the emulator is', base_path_emulator)
    model_files = [
        f for f in os.listdir(base_path_emulator)
        if f.startswith("last_model_foldno_") and f.endswith("_120x3.pth")
    ]
    print('Check model_files', model_files)

    if len(model_files) == 1:
        # Check for single-fold model
        single_model_path = os.path.join(base_path_emulator, "last_model_foldno_train_120x3.pth")
        if os.path.exists(single_model_path):
            print(f"Loading single model: {single_model_path}")
            model.model.load_state_dict(torch.load(single_model_path, map_location=torch.device('cpu')))
            model.model.eval()  # Set model to evaluation mode
            models.append(model)
    return min_in, max_in, models

def get_best_fit(name_chain, samples, base_path, post, irsic, kmax, noise, use_noise):
    """
    Calls get_models_z to load the models from the base_path given. 
    Uses fiducial covariance matrix.  
    """

    bc_cdm_paths = {
        4.2: "/data/og313b/lyalpha/models/cdm/planck1_20_1024/bc_4.2_1.0.txt",
        4.6: "/data/og313b/lyalpha/models/cdm/planck1_20_1024/bc_4.6_1.0.txt",
        5.0: "/data/og313b/lyalpha/models/cdm/planck1_20_1024/bc_5.0_1.0.txt"
    }
    # Fiducial analysis no RC applies a post correction:L
    rc_cdm_paths = {
        4.2: "/data/og313b/lyalpha/models/cdm/planck1_20_1024/Rs10_4.2_1.0.txt",
        4.6: "/data/og313b/lyalpha/models/cdm/planck1_20_1024/Rs10_4.6_1.0.txt",
        5.0: "/data/og313b/lyalpha/models/cdm/planck1_20_1024/Rs10_5.0_1.0.txt"
    }

    if irsic == True:
        # arr = [0.83, 0.82, 0.94, 1.13, 1.06, 1.01, 1.11, 1.43, 1.59, 6.21, 5.95, 3.49, 0.05, 1]
        bestfit, onesigma_all = get_bf_sigma_irsic_all(name_chain, samples)
        arr = bestfit
        print('Best-fit samples FOR IRSIC+24', arr)
        
    elif irsic == False:
        if noise == False:
            # Get best-fit parameters for samples considered:
            like_stats = samples.getLikeStats()
            n_pars = 14
            arr = [like_stats.names[i].bestfit_sample for i in range(n_pars)]
            print('Best-fit samples', arr)

        elif noise ==True:
            like_stats = samples.getLikeStats()
            #stats = samples.getMargeStats()
            n_pars = 17
            #arr = [stats.names[i].mean for i in range(n_pars)]
            arr = [like_stats.names[i].bestfit_sample for i in range(n_pars)]
            print('Best-fit samples', arr)
    
    min_in42, max_in42, models42 = get_models_z(4.2, base_path)
    min_in46, max_in46, models46 = get_models_z(4.6, base_path)
    min_in5, max_in5, models5 = get_models_z(5.0, base_path)

    log_k, P_kRC42, P_kRC46, P_kRC5, boera_covariance_matrix = get_data()

    # Depening on interpolation method, load necessary variables:
    # Define column names
    if post ==True:
        column_names = ['Model', 'zr', 'tau_e', 'u0_4.2<=z<=12.0', 'u0_4.6<=z<=13.0', 'u0_6.0<=z<=13.0']

        # Load data from text file, skipping the first 5 rows
        u0all = pd.read_csv('/data/og313b/lyalpha/taue_u0_table.txt', skiprows=4, delim_whitespace=True, names=column_names)
        models_cosmo = ['planck1_20_1024', 'planck1_20_1024_cold', 'planck1_20_1024_hot', 'planck1_20_1024_zr525_cold', 'planck1_20_1024_zr525', 'planck1_20_1024_zr675', 'planck1_20_1024_zr750']

        # Reduce table to models for which we have u0 in wdm cosmlogies:
        filtered_u0all = u0all[u0all['Model'].isin(models_cosmo)].sort_values('u0_4.2<=z<=12.0').reset_index(drop=True)

        # Models in increasing u0 values for each redshift:
        u042 = filtered_u0all.iloc[:,3].reset_index(drop=True)
        u046 = filtered_u0all.iloc[:,4].reset_index(drop=True)
        u05 = filtered_u0all.iloc[:,5].reset_index(drop=True)


        Rs10_42 = np.load('/data/og313/lyalpha/code_wdm/analysis_RC_cdm_u0_post/Rs10_42.npy')
        Rs10_46 = np.load('/data/og313/lyalpha/code_wdm/analysis_RC_cdm_u0_post/Rs10_46.npy')
        Rs10_5 = np.load('/data/og313/lyalpha/code_wdm/analysis_RC_cdm_u0_post/Rs10_5.npy')

        print(np.shape(u042), np.shape(Rs10_42))

    if noise == True:
        k = 10**log_k
        # If noise is added:
        #Window function eq. (7) in Boera19:
        R =  2.55 #km/s, nominal resolution #compared to R = 60 for paper: https://arxiv.org/pdf/1306.5896.pdf
        dv = 2.5 #km/s, pixel size
        def window_function(k):
            return np.exp(-((k*R)**2)/2)*((np.sin(k*dv/2))/(k*dv/2)) #eq. (7) in Boera19
        
        W = window_function(k)
        mean_noise_params = np.array([0.08, 0.1, 0.12])


    if kmax ==True:

        #kmax cut analysis: re-define covariance matrix:                                             
        block1 = boera_covariance_matrix[:16, :16]
        block2 = boera_covariance_matrix[16:32, 16:32]
        block3 = boera_covariance_matrix[32:48, 32:48]

        # Remove the last three rows and columns from each (16,16) block to get (13,13) blocks       
        block1_reduced = block1[:-3, :-3]
        block2_reduced = block2[:-3, :-3]
        block3_reduced = block3[:-3, :-3]

        # Concatenate the resulting (13,13) blocks to form the final (39,39) covariance matrix       
        covariance_matrix = np.block([
            [block1_reduced, np.zeros((13, 13)), np.zeros((13, 13))],
            [np.zeros((13, 13)), block2_reduced, np.zeros((13, 13))],
            [np.zeros((13, 13)), np.zeros((13, 13)), block3_reduced]
        ])

        print('Shape of the covariance matrix', np.shape(covariance_matrix))

        # shape = boera_covariance_matrix.shape
        # updated_matrix = np.zeros(shape)
        # indices = [-3, -2, -1]

        # # Function to update only the diagonal elements in a given block
        # def update_diagonal(matrix, block_start, block_end):
        #     for i in indices:
        #         original_value = boera_covariance_matrix[block_end + i, block_end + i]
        #         updated_value = original_value * 0.50  # Get 50% of the sigma^2 in last three bins
        #         matrix[block_end + i, block_end + i] = updated_value

        # # Apply the function to the three blocks
        # update_diagonal(updated_matrix, 0, 16)   # Block1
        # update_diagonal(updated_matrix, 16, 32)  # Block2
        # update_diagonal(updated_matrix, 32, 48)  # Block3

        # covariance_matrix = updated_matrix + boera_covariance_matrix
        # print(np.diag(covariance_matrix)[13:16], np.diag(covariance_matrix)[29:32], np.diag(covariance_matrix)[45:48])

    elif kmax == False:
        covariance_matrix =boera_covariance_matrix


    print('SHAPE OF COVARIANCE MATRIX USED', np.shape(covariance_matrix))


    if irsic == True:
        if noise == True:
            print('Parameter use as alpha', arr[-4])
            an_z42 = arr[-3]
            bf_input_z42 = [arr[0], arr[3], arr[6], arr[9], arr[-4], 1]
        elif noise == False:
            bf_input_z42 = [arr[0], arr[3], arr[6], arr[9], arr[-2], 1] #Pass emulator with fwdm = 1
        print('Best-fit values used at z = 4.2',  bf_input_z42 )
    if irsic == False:
        if noise == True:
            #print('best-fit input without noise:')
            bf_input_z42 = [arr[0], arr[3], arr[6], arr[9], arr[-2], arr[-1]]
            an_z42 = arr[-5]
            print('Noise param at rz = 4.2', an_z42)
            print('best-fit params without noise at z = 4.2', bf_input_z42)
        else:
            bf_input_z42 = [arr[0], arr[3], arr[6], arr[9], arr[-2], arr[-1]]
    print(bf_input_z42 )
    # Choose interpolation method:
    if post ==True:
        bf_pred_42 = get_nn_prediction_post_rs_mcmc(4.2, bf_input_z42 , min_in42, max_in42, models42, u042, u046, u05, Rs10_42, Rs10_46, Rs10_5)
    elif post==False:
        if noise == True:
            if use_noise == True:
                bc_cdm_path = bc_cdm_paths[4.2]
                bc_cdm = np.loadtxt(bc_cdm_path)

                rc_cdm_path = rc_cdm_paths[4.2]
                rc_cdm = np.loadtxt(rc_cdm_path)
                pred_emu = get_nn_prediction_mcmc_noise(4.2, bf_input_z42 , min_in42, max_in42, models42)
                pred_total = pred_emu + (an_z42*0.08*np.ones(16)/(W**2))
                bf_pred_42 = pred_total * bc_cdm * rc_cdm
            if use_noise == False:
                bc_cdm_path = bc_cdm_paths[4.2]
                bc_cdm = np.loadtxt(bc_cdm_path)

                rc_cdm_path = rc_cdm_paths[4.2]
                rc_cdm = np.loadtxt(rc_cdm_path)
                pred_emu = get_nn_prediction_mcmc_noise(4.2, bf_input_z42 , min_in42, max_in42, models42)
                pred_total = pred_emu #+ (an_z42*0.08*np.ones(16)/(W**2))
                #print('How much contribution would give the noise term', (an_z42*0.08*np.ones(16)/(W**2)))
               # print('How much contribution the noise term with resolution correction',  bc_cdm * rc_cdm*(an_z42*0.08*np.ones(16)/(W**2)))
                bf_pred_42 = pred_total * bc_cdm * rc_cdm
                
        elif noise == False:
            bf_pred_42 = get_nn_prediction_mcmc(4.2, bf_input_z42, min_in42, max_in42, models42, rc=True)

   # if noise == True:
   #     if use_noise == True:
   #         print('Add extra prediction to emulator')
   #         bf_pred_42 = bf_pred_42 + an_z42*mean_noise_params[0]*np.ones(16)/((W**2))
   #     elif use_noise == False:
   #         print('No extra prediction added to emulator')
   #         bf_pred_42 = bf_pred_42 
            # print('best fit predicted power spectra', bf_pred_42 )

    bf_pred_42 = np.array(bf_pred_42).flatten()
    if kmax == True:
        residuals = bf_pred_42[:-3] - P_kRC42[:-3]
        cov_matrix = covariance_matrix[:13, :13]
        print('Shape of residuals', residuals)
    else:
        residuals = bf_pred_42 - P_kRC42
        cov_matrix = covariance_matrix[:16, :16]

    chi42_value = compute_chi_squared(residuals.reshape(-1,1), cov_matrix)
    print(f"Chi-squared: {chi42_value}")

    
    if irsic == True:
        if noise == True:
            print('Parameter used as alpha', arr[-4])
            an_z46 = arr[-2]
            print('Parameter used as A noise z = 4.6', arr[-2])
            bf_input_z46 = [arr[1], arr[4], arr[7], arr[10], arr[-4], 1]
        elif noise == False:
            bf_input_z46 = [arr[1], arr[4], arr[7], arr[10], arr[-2], 1]
        print('Best-fit values used at z = 4.6',  bf_input_z46)

    if irsic == False:
        if noise == True:
            bf_input_z46 = [arr[1], arr[4], arr[7], arr[10], arr[-2], arr[-1]]
            an_z46 = arr[-4]
            print('Noise param at z = 4.6', an_z46)
            print('best-fit params with noise at z = 4.6', bf_input_z46)
        else: 
            bf_input_z46 = [arr[1], arr[4], arr[7], arr[10], arr[-2], arr[-1]]
    if post == True:
        bf_pred_46 = get_nn_prediction_post_rs_mcmc(4.6, bf_input_z46, min_in46, max_in46, models46, u042, u046, u05, Rs10_42, Rs10_46, Rs10_5)
    elif post==False:
       # bf_pred_46 = get_nn_prediction_mcmc(4.6, bf_input_z46, min_in46, max_in46, models46, rc=True)
    
        if noise == True:
            if use_noise == True:
                bc_cdm_path = bc_cdm_paths[4.6]
                bc_cdm = np.loadtxt(bc_cdm_path)

                rc_cdm_path = rc_cdm_paths[4.6]
                rc_cdm = np.loadtxt(rc_cdm_path)
                pred_emu = get_nn_prediction_mcmc_noise(4.6, bf_input_z46 , min_in46, max_in46, models46)
                pred_total = pred_emu + (an_z46*0.1*np.ones(16)/(W**2))
                bf_pred_46 = pred_total * bc_cdm * rc_cdm
            if use_noise == False:
                bc_cdm_path = bc_cdm_paths[4.6]
                bc_cdm = np.loadtxt(bc_cdm_path)

                rc_cdm_path = rc_cdm_paths[4.6]
                rc_cdm = np.loadtxt(rc_cdm_path)
                pred_emu = get_nn_prediction_mcmc_noise(4.6, bf_input_z46 , min_in46, max_in46, models46)
                pred_total = pred_emu #+ (an_z42*0.08*np.ones(16)/(W**2))
                #print('How much contribution would give the noise term', (an_z42*0.08*np.ones(16)/(W**2)))
               # print('How much contribution the noise term with resolution correction',  bc_cdm * rc_cdm*(an_z42*0.
                bf_pred_46 = pred_total * bc_cdm * rc_cdm

        if noise == False:
             bf_pred_46 = get_nn_prediction_mcmc(4.6, bf_input_z46, min_in46, max_in46, models46, rc=True)
    bf_pred_46 = np.array(bf_pred_46).flatten()
    
    if kmax == True:
        residuals = bf_pred_46[:-3] - P_kRC46[:-3]
        #print('Shape of residuals', residuals)
        cov_matrix = covariance_matrix[13:26, 13:26]
    else:
        residuals = bf_pred_46 - P_kRC46
        cov_matrix = covariance_matrix[16:32, 16:32]

    chi46_value = compute_chi_squared(residuals.reshape(-1,1), cov_matrix)

    print(f"Chi-squared: {chi46_value}")

    if irsic == True:
        if noise == True:
            an_z5 = arr[-1]
            bf_input_z5 = [arr[2], arr[5], arr[8], arr[11], arr[-4], 0]
        elif noise == False:
            bf_input_z5 = [arr[2], arr[5], arr[8], arr[11], arr[-1], 0]
        print('Best-fit values used at z = 5',  bf_input_z5 )


    if irsic == False:
        if noise == True:
            bf_input_z5 = [arr[2], arr[5], arr[8], arr[11], arr[-2], arr[-1]]
            an_z5 = arr[-3]
            print('Best fit noise param at z = 5', an_z5)
            print('best-fit params with noise at z = 5.0', bf_input_z5)

        else: 
            bf_input_z5 = [arr[2], arr[5], arr[8], arr[11], arr[-2], arr[-1]]
    if post==True:
        bf_pred_5 = get_nn_prediction_post_rs_mcmc(5.0, bf_input_z5, min_in5, max_in5, models5, u042, u046, u05, Rs10_42, Rs10_46, Rs10_5)
    elif post==False:
        if noise == True:
            if use_noise == True:
                bc_cdm_path = bc_cdm_paths[5.0]
                bc_cdm = np.loadtxt(bc_cdm_path)

                rc_cdm_path = rc_cdm_paths[5.0]
                rc_cdm = np.loadtxt(rc_cdm_path)
                pred_emu = get_nn_prediction_mcmc_noise(5.0, bf_input_z5 , min_in5, max_in5, models5)
                pred_total = pred_emu + (an_z5*0.12*np.ones(16)/(W**2))
                bf_pred_5 = pred_total * bc_cdm * rc_cdm
            if use_noise == False:
                bc_cdm_path = bc_cdm_paths[5.0]
                bc_cdm = np.loadtxt(bc_cdm_path)

                rc_cdm_path = rc_cdm_paths[5.0]
                rc_cdm = np.loadtxt(rc_cdm_path)
                pred_emu = get_nn_prediction_mcmc_noise(5.0, bf_input_z5 , min_in5, max_in5, models5)
                pred_total = pred_emu #+ (an_z42*0.08*np.ones(16)/(W**2))
                #print('How much contribution would give the noise term', (an_z42*0.08*np.ones(16)/(W**2)))
               # print('How much contribution the noise term with resolution correction',  bc_cdm * rc_cdm*(an_z42*0.08*np.ones(16)/(W**2)))
                bf_pred_5 = pred_total * bc_cdm * rc_cdm


        if noise ==False:
            bf_pred_5 = get_nn_prediction_mcmc(5.0, bf_input_z5, min_in5, max_in5, models5, rc=True)
    
    bf_pred_5 = np.array(bf_pred_5).flatten()
    if kmax == True:
        residuals = bf_pred_5[:-3] - P_kRC5[:-3]
        cov_matrix = covariance_matrix[26:, 26:]
    else:
        residuals = bf_pred_5 - P_kRC5
        cov_matrix = covariance_matrix[32:, 32:]

    chi5_value = compute_chi_squared(residuals.reshape(-1,1), cov_matrix)

    print(f"Chi-squared: {chi5_value}")

    bf_pred_5 = np.array(bf_pred_5).flatten()
    bf_pred_46 = np.array(bf_pred_46).flatten()
    bf_pred_42 = np.array(bf_pred_42).flatten()


    # FOR TABLE:
    if noise == True:
          print('BEST FIT VALUES TO PUT IN TABLE', bf_input_z46[0], bf_input_z46[3], bf_input_z46[4])
          comb_chi = chi42_value + chi46_value + chi5_value
          print('COMBINED chi squared is', comb_chi)
    else:
        print('BEST FIT VALUES TO PUT IN TABLE', bf_input_z46[0], bf_input_z46[3],1/ bf_input_z46[-2], bf_input_z46[-1])
        comb_chi = chi42_value + chi46_value + chi5_value
        print('COMBINED chi squared is', comb_chi)

    
    # Get 1 sigma constraints:
    params = ['T0z46', 'u0z46']
    stats = samples.getMargeStats()
    for i in params:
        lims = stats.parWithName(f'{i}').limits
        for conf, lims in zip(samples.contours,lims):
            if conf == 0.68:
                if i == 'T0z46':
                    print(f'{i}:', (bf_input_z46[0]/(10**4) - lims.lower, lims.upper- bf_input_z46[0]/(10**4)))
                elif i == 'u0z46':
                    print(f'{i}:', (bf_input_z46[3] - lims.lower, lims.upper - bf_input_z46[3]))

    if noise == True:
        params = ['Anz46']
        stats = samples.getMargeStats()
        for i in params:
            lims = stats.parWithName(f'{i}').limits
            for conf, lims in zip(samples.contours,lims):
                if conf == 0.68:
                    print(f'{i}:', (an_z46 - lims.lower, lims.upper- an_z46))
                    print(f'{i}:', (lims.lower, lims.upper))
                
        ### more accurately                                                                                                                                                                  
    two_sigma = 1-0.954499736103642                                                                    #                
    three_sigma = 1-0.997300203936740                                                                                     
    four_sigma = 1-0.999936657516334# = 4sigma                                                                                       
    five_sigma = 1-0.999999426696856 #= 5sigma

    upper_bound_twosig = (samples.confidence('alpha', limfrac=two_sigma, upper=True))
    print(upper_bound_twosig)
    print('mwdm <', 1/upper_bound_twosig, 'keV  is excluded at 2 sigma')

    #In case u0 is also very skewed:
    upper_bound_twosig = (samples.confidence('u0z46', limfrac=two_sigma, upper=True))
    print('u0_z46 <', upper_bound_twosig, 'eV/mp  is excluded at 2 sigma')
    if irsic == False:
        upper_bound_twosig = (samples.confidence('fwdm', limfrac=two_sigma, upper=True))
        print('fwdm <', upper_bound_twosig, 'is excluded at 2 sigma')
    

    return chi42_value, chi46_value, chi5_value, bf_pred_42.flatten(), bf_pred_46.flatten(), bf_pred_5.flatten()

def plot_best_fit_vs_irsic(bf_pred_42, bf_pred_46, bf_pred_5, chi42_value, chi46_value, chi5_value, 
                           bf_pred_42_other, bf_pred_46_other, bf_pred_5_other, chi42_value_other, chi46_value_other, chi5_value_other, save_path, analysis, kmax):

    """
    Plot best-fit against Irsic+24 including chi^2 value.
    _other corresponds to Irsic+24
    """

    #Get data:
    log_k, P_kRC42, P_kRC46, P_kRC5, boera_covariance_matrix = get_data()
    orig_covariance_matrix = boera_covariance_matrix.copy()
    if kmax == True:
        shape = boera_covariance_matrix.shape
        updated_matrix = np.zeros(shape)
        indices = [-3, -2, -1]

        # Function to update only the diagonal elements in a given block
        def update_diagonal(matrix, block_start, block_end):
            for i in indices:
                original_value = boera_covariance_matrix[block_end + i, block_end + i]
                updated_value = original_value * 0.50  # Get 50% of the sigma^2 in last three bins
                matrix[block_end + i, block_end + i] = updated_value

        # Apply the function to the three blocks
        update_diagonal(updated_matrix, 0, 16)   # Block1
        update_diagonal(updated_matrix, 16, 32)  # Block2
        update_diagonal(updated_matrix, 32, 48)  # Block3

        covariance_matrix = updated_matrix + boera_covariance_matrix
        print(np.diag(covariance_matrix)[13:16], np.diag(covariance_matrix)[29:32], np.diag(covariance_matrix)[45:48])

    elif kmax == False:
        covariance_matrix =boera_covariance_matrix
        print(np.diag(covariance_matrix)[13:16], np.diag(covariance_matrix)[29:32], np.diag(covariance_matrix)[45:48])
    # Plot best-fit value to check how it looks like:
    fig, ax = plt.subplots(2,3, sharex = True, figsize = (50, 20))

    # Create a 2x3 grid
    plt.rcParams.update({'font.size': 35})

    #Create a 2x3 grid with different height ratios (top rows are 3x bigger than bottom rows)
    gs = gridspec.GridSpec(2, 3, height_ratios=[3/2, 1/2], width_ratios=[3, 3, 3])
    # Create the subplots
    ax = [[plt.subplot(gs[i, j]) for j in range(3)] for i in range(2)]


    logk = np.arange(-2.2,-0.7, 0.1)
    k = 10**logk

    ax[0][0].plot(log_k, bf_pred_42* k / np.pi, linestyle='--',  label=f'z = 4.2, $\chi^{{2}} = {round(chi42_value.item(), 2)}$, {analysis}', markersize=6,color='tab:blue')
    ax[0][0].plot(log_k, bf_pred_42_other* k / np.pi, linestyle='--',  label=f'$\chi^{{2}} = {round(chi42_value_other.item(), 2)}$', markersize=6,color='black')
    ax[0][0].errorbar(log_k, P_kRC42 * k / np.pi, label='Boera+19', yerr=(k/ np.pi)*np.sqrt(covariance_matrix[:16, :16].diagonal()), fmt='D', markersize=6, color='tab:blue')
    
    ax[0][0].set_ylabel('kP(k)$/\pi$')
    ax[0][0].set_xlabel('$\mathrm{log}_{10}$ ($k$ $\mathrm{[km^{-1}\,s])}$')
    ax[0][0].set_ylim(0.01,0.8)
    ax[0][0].set_xlim(-2.25, -0.65)
    ax[0][0].set_yscale('log')
    ax[0][0].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax[0][0].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax[0][0].legend(frameon=False)

    ax[1][0].set_aspect('auto', adjustable='box', anchor='C')
    ax[1][0].errorbar(log_k, P_kRC42/bf_pred_42, yerr=np.sqrt(covariance_matrix[:16, :16].diagonal())/bf_pred_42,  fmt='D', color='tab:blue', markersize=6)  
    ax[1][0].errorbar(log_k, P_kRC42/bf_pred_42_other, yerr=np.sqrt(orig_covariance_matrix[:16, :16].diagonal())/bf_pred_42_other,  fmt='D', color='black', markersize=6) 
    ax[1][0].axhline(y=1, linestyle='--',color = 'tab:grey')
    ax[1][0].set_ylabel('Residual')
    ax[1][0].set_xlabel('$\mathrm{log}_{10}$ ($k$ $\mathrm{[km^{-1}\,s])}$')
    ax[1][0].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax[1][0].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax[1][0].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax[1][0].set_yticks([0.5, 1.0, 1.5])
    ax[1][0].legend()

    # Repeat similar setup for z = 4.6
    ax[0][1].plot(log_k, bf_pred_46* k / np.pi, linestyle='--', color='tab:orange', label=f'z = 4.6.0, $\chi^{{2}} = {round(chi46_value.item(), 2)}$')
    ax[0][1].plot(log_k, bf_pred_46_other* k / np.pi, linestyle='--', color='black', label=f' $\chi^{{2}} = {round(chi46_value_other.item(), 2)}$, Irsic+24')
    # ax[0][1].plot(log_k, bf_pred_vid_46* k / np.pi, linestyle='--', color='red', label=f'$\chi^{{2}} = {round(chi2_46_vid, 2)}$')
    ax[0][1].errorbar(log_k, P_kRC46 * k / np.pi,  yerr=(k/ np.pi)*np.sqrt(covariance_matrix[16:32, 16:32].diagonal()), fmt='D', markersize=6, color='tab:orange')
    ax[0][1].set_ylabel('kP(k)$/\pi$')
    ax[0][1].set_xlabel('$\mathrm{log}_{10}$ ($k$ $\mathrm{[km^{-1}\,s])}$')
    ax[0][1].set_ylim(0.01,0.8)
    ax[0][1].set_xlim(-2.25, -0.65)
    ax[0][1].set_yscale('log')
    ax[0][1].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax[0][1].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax[0][1].legend(frameon=False)

    ax[1][1].set_aspect('auto', adjustable='box', anchor='C')
    ax[1][1].errorbar(log_k, P_kRC46/bf_pred_46, yerr=np.sqrt(covariance_matrix[16:32, 16:32].diagonal())/bf_pred_46, fmt='D', color='tab:orange', markersize=6) 
    ax[1][1].errorbar(log_k, P_kRC46/bf_pred_46_other, yerr=np.sqrt(orig_covariance_matrix[16:32, 16:32].diagonal())/bf_pred_46_other, fmt='D', color='black', markersize=6)
    ax[1][1].axhline(y=1, linestyle='--',color = 'tab:grey')
    ax[1][1].set_ylabel('Residual')
    ax[1][1].set_xlabel('$\mathrm{log}_{10}$ ($k$ $\mathrm{[km^{-1}\,s])}$')
    ax[1][1].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax[1][1].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax[1][1].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax[1][1].set_yticks([0.5, 1.0, 1.5])
    ax[1][1].legend()


    # Repeat similar setup for z = 5.0
    ax[0][2].plot(log_k, bf_pred_5* k / np.pi, linestyle='--', color='tab:green', label=f'z = 5.0, $\chi^{{2}} = {round(chi5_value.item(), 2)}$')
    ax[0][2].plot(log_k, bf_pred_5_other* k / np.pi, linestyle='--', color='black', label=f' $\chi^{{2}} = {round(chi5_value_other.item(), 2)}$')
    ax[0][2].errorbar(log_k, P_kRC5 * k / np.pi,  yerr=(k/ np.pi)*np.sqrt(covariance_matrix[32:, 32:].diagonal()), fmt='D', markersize=6, color='tab:green')

    ax[0][2].set_ylabel('kP(k)$/\pi$')
    ax[0][2].set_xlabel('$\mathrm{log}_{10}$ ($k$ $\mathrm{[km^{-1}\,s])}$')
    ax[0][2].set_ylim(0.01,0.8)
    ax[0][2].set_xlim(-2.25, -0.65)
    ax[0][2].set_yscale('log')
    ax[0][2].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax[0][2].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax[0][2].legend(frameon=False)


    ax[1][2].set_aspect('auto', adjustable='box', anchor='C')
    ax[1][2].errorbar(log_k, P_kRC5/bf_pred_5_other, yerr = np.sqrt(orig_covariance_matrix[32:, 32:].diagonal())/bf_pred_5_other, fmt='D', markersize=6,color='black') 
    ax[1][2].errorbar(log_k, P_kRC5/bf_pred_5, yerr = np.sqrt(covariance_matrix[32:, 32:].diagonal())/bf_pred_5, fmt='D', markersize=6, color='tab:green') 
    ax[1][2].axhline(y=1, linestyle='--',color = 'tab:grey')
    ax[1][2].set_ylabel('Residual')
    ax[1][2].set_xlabel('$\mathrm{log}_{10}$ ($k$ $\mathrm{[km^{-1}\,s])}$')
    ax[1][2].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax[1][2].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax[1][2].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax[1][2].set_yticks([0.5, 1.0, 1.5])
    ax[1][2].legend()

    if save_path != None:
        plt.savefig(save_path, bbox_inches='tight') 
    plt.subplots_adjust(right=0.95)
    plt.show()


def plot_best_fit(bf_pred_42, bf_pred_46, bf_pred_5, chi42_value, chi46_value, chi5_value, 
                        save_path, analysis, kmax):

    """
    Plot best-fit including chi^2 value.
    """

    #Get data:
    log_k, P_kRC42, P_kRC46, P_kRC5, boera_covariance_matrix = get_data()
    orig_covariance_matrix = boera_covariance_matrix.copy()
    if kmax == True:
        shape = boera_covariance_matrix.shape
        updated_matrix = np.zeros(shape)
        indices = [-3, -2, -1]

        # Function to update only the diagonal elements in a given block
        def update_diagonal(matrix, block_start, block_end):
            for i in indices:
                original_value = boera_covariance_matrix[block_end + i, block_end + i]
                updated_value = original_value * 0.50  # Get 50% of the sigma^2 in last three bins
                matrix[block_end + i, block_end + i] = updated_value

        # Apply the function to the three blocks
        update_diagonal(updated_matrix, 0, 16)   # Block1
        update_diagonal(updated_matrix, 16, 32)  # Block2
        update_diagonal(updated_matrix, 32, 48)  # Block3

        covariance_matrix = updated_matrix + boera_covariance_matrix
        print(np.diag(covariance_matrix)[13:16], np.diag(covariance_matrix)[29:32], np.diag(covariance_matrix)[45:48])

    elif kmax == False:
        covariance_matrix =boera_covariance_matrix
        print(np.diag(covariance_matrix)[13:16], np.diag(covariance_matrix)[29:32], np.diag(covariance_matrix)[45:48])
    # Plot best-fit value to check how it looks like:
    fig, ax = plt.subplots(2,3, sharex = True, figsize = (45, 19))

    # Create a 2x3 grid
    plt.rcParams.update({'font.size': 37})

    #Create a 2x3 grid with different height ratios (top rows are 3x bigger than bottom rows)
    gs = gridspec.GridSpec(2, 3, height_ratios=[3/2, 1/2], width_ratios=[3, 3, 3], wspace=0.25)
    # Create the subplots
    ax = [[plt.subplot(gs[i, j]) for j in range(3)] for i in range(2)]
    line_width_ax =1
    line_width = 3.5
    markersize = 7
    
    logk = np.arange(-2.2,-0.7, 0.1)
    k = 10**logk

    line_width = 2

    ax[0][0].plot(log_k, bf_pred_42* k / np.pi, linestyle='--',  label=f'z = 4.2, $\chi^{{2}} = {round(chi42_value.item(), 2)}$, {analysis}', markersize=markersize,color='tab:blue',  linewidth=line_width)
    #ax[0][0].plot(log_k, bf_pred_42_other* k / np.pi, linestyle='--',  label=f'$\chi^{{2}} = {round(chi42_value_other.item(), 2)}$', markersize=6,color='black')
    ax[0][0].errorbar(log_k, P_kRC42 * k / np.pi, label='Boera+19', yerr=(k/ np.pi)*np.sqrt(covariance_matrix[:16, :16].diagonal()), fmt='D', markersize=markersize, color='tab:blue',  linewidth=line_width)
    
    ax[0][0].set_ylabel('kP(k)$/\pi$')
    #ax[0][0].set_xlabel('$\mathrm{log}_{10}$ ($k$ $\mathrm{[km^{-1}\,s])}$')
    ax[0][0].set_ylim(0.01,0.8)
    ax[0][0].set_xlim(-2.25, -0.65)
    ax[0][0].set_yscale('log')
    ax[0][0].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax[0][0].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax[0][0].legend(frameon=False, loc = 'lower left')

    
    # Set the line width for the axes
    ax[0][0].spines['top'].set_linewidth(line_width_ax)
    ax[0][0].spines['right'].set_linewidth(line_width_ax)
    ax[0][0].spines['bottom'].set_linewidth(line_width_ax)
    ax[0][0].spines['left'].set_linewidth(line_width_ax)

    ax[1][0].set_aspect('auto', adjustable='box', anchor='C')
    ax[1][0].errorbar(log_k, P_kRC42/bf_pred_42, yerr=np.sqrt(covariance_matrix[:16, :16].diagonal())/bf_pred_42,  fmt='D', color='tab:blue', markersize=markersize, linewidth = line_width)
    #ax[1][0].errorbar(log_k, P_kRC42/bf_pred_42_other, yerr=np.sqrt(orig_covariance_matrix[:16, :16].diagonal())/bf_pred_42_other,  fmt='D', color='black', markersize=6) 
    ax[1][0].axhline(y=1, linestyle='--',color = 'tab:grey')
    ax[1][0].set_ylabel('Residual')
    #ax[1][0].set_xlabel('$\mathrm{log}_{10}$ ($k$ $\mathrm{[km^{-1}\,s])}$')
    ax[1][0].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax[1][0].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax[1][0].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax[1][0].set_yticks([0.5, 1.0, 1.5])
    ax[1][0].set_xlabel('$\mathrm{log}_{10}$ ($k$ $\mathrm{[km^{-1}\,s])}$')
    ax[1][0].legend()

    ax[1][0].spines['top'].set_linewidth(line_width_ax)
    ax[1][0].spines['right'].set_linewidth(line_width_ax)
    ax[1][0].spines['bottom'].set_linewidth(line_width_ax)
    ax[1][0].spines['left'].set_linewidth(line_width_ax)


    # Repeat similar setup for z = 4.6
    ax[0][1].plot(log_k, bf_pred_46* k / np.pi, linestyle='--', color='tab:orange', label=f'z = 4.6.0, $\chi^{{2}} = {round(chi46_value.item(), 2)}$', linewidth = line_width)
    #ax[0][1].plot(log_k, bf_pred_46_other* k / np.pi, linestyle='--', color='black', label=f' $\chi^{{2}} = {round(chi46_value_other.item(), 2)}$, Irsic+24')
    # ax[0][1].plot(log_k, bf_pred_vid_46* k / np.pi, linestyle='--', color='red', label=f'$\chi^{{2}} = {round(chi2_46_vid, 2)}$')
    ax[0][1].errorbar(log_k, P_kRC46 * k / np.pi,  yerr=(k/ np.pi)*np.sqrt(covariance_matrix[16:32, 16:32].diagonal()), fmt='D', markersize=markersize, linewidth = line_width, color='tab:orange')
    ax[0][1].set_ylabel('kP(k)$/\pi$')
    #ax[0][1].set_xlabel('$\mathrm{log}_{10}$ ($k$ $\mathrm{[km^{-1}\,s])}$')
    ax[0][1].set_ylim(0.01,0.8)
    ax[0][1].set_xlim(-2.25, -0.65)
    ax[0][1].set_yscale('log')
    ax[0][1].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax[0][1].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax[0][1].legend(frameon=False)

    ax[0][1].spines['top'].set_linewidth(line_width_ax)
    ax[0][1].spines['right'].set_linewidth(line_width_ax)
    ax[0][1].spines['bottom'].set_linewidth(line_width_ax)
    ax[0][1].spines['left'].set_linewidth(line_width_ax)

    ax[1][1].set_aspect('auto', adjustable='box', anchor='C')
    ax[1][1].errorbar(log_k, P_kRC46/bf_pred_46, yerr=np.sqrt(covariance_matrix[16:32, 16:32].diagonal())/bf_pred_46, fmt='D', color='tab:orange', markersize=markersize, linewidth = line_width)
    #ax[1][1].errorbar(log_k, P_kRC46/bf_pred_46_other, yerr=np.sqrt(orig_covariance_matrix[16:32, 16:32].diagonal())/bf_pred_46_other, fmt='D', color='black', markersize=6)
    ax[1][1].axhline(y=1, linestyle='--',color = 'tab:grey')
    ax[1][1].set_ylabel('Residual')
    #ax[1][1].set_xlabel('$\mathrm{log}_{10}$ ($k$ $\mathrm{[km^{-1}\,s])}$')
    ax[1][1].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax[1][1].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax[1][1].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax[1][1].set_yticks([0.5, 1.0, 1.5])
    ax[1][1].set_xlabel('$\mathrm{log}_{10}$ ($k$ $\mathrm{[km^{-1}\,s])}$')
    ax[1][1].legend()

    ax[1][1].spines['top'].set_linewidth(line_width_ax)
    ax[1][1].spines['right'].set_linewidth(line_width_ax)
    ax[1][1].spines['bottom'].set_linewidth(line_width_ax)
    ax[1][1].spines['left'].set_linewidth(line_width_ax)



    # Repeat similar setup for z = 5.0
    ax[0][2].plot(log_k, bf_pred_5* k / np.pi, linestyle='--', color='tab:green', label=f'z = 5.0, $\chi^{{2}} = {round(chi5_value.item(), 2)}$', linewidth = line_width)
    #ax[0][2].plot(log_k, bf_pred_5_other* k / np.pi, linestyle='--', color='black', label=f' $\chi^{{2}} = {round(chi5_value_other.item(), 2)}$')
    ax[0][2].errorbar(log_k, P_kRC5 * k / np.pi,  yerr=(k/ np.pi)*np.sqrt(covariance_matrix[32:, 32:].diagonal()), fmt='D', markersize=markersize, color='tab:green', linewidth = line_width)

    ax[0][2].set_ylabel('kP(k)$/\pi$')
    #ax[0][2].set_xlabel('$\mathrm{log}_{10}$ ($k$ $\mathrm{[km^{-1}\,s])}$')
    ax[0][2].set_ylim(0.01,0.8)
    ax[0][2].set_xlim(-2.25, -0.65)
    ax[0][2].set_yscale('log')
    ax[0][2].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax[0][2].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax[0][2].legend(frameon=False)

    ax[0][2].spines['top'].set_linewidth(line_width_ax)
    ax[0][2].spines['right'].set_linewidth(line_width_ax)
    ax[0][2].spines['bottom'].set_linewidth(line_width_ax)
    ax[0][2].spines['left'].set_linewidth(line_width_ax)



    ax[1][2].set_aspect('auto', adjustable='box', anchor='C')
    #ax[1][2].errorbar(log_k, P_kRC5/bf_pred_5_other, yerr = np.sqrt(orig_covariance_matrix[32:, 32:].diagonal())/bf_pred_5_other, fmt='D', markersize=6,color='black') 
    ax[1][2].errorbar(log_k, P_kRC5/bf_pred_5, yerr = np.sqrt(covariance_matrix[32:, 32:].diagonal())/bf_pred_5, fmt='D', markersize=markersize, color='tab:green', linewidth = line_width)
    ax[1][2].axhline(y=1, linestyle='--',color = 'tab:grey')
    ax[1][2].set_ylabel('Residual')
    #ax[1][2].set_xlabel('$\mathrm{log}_{10}$ ($k$ $\mathrm{[km^{-1}\,s])}$')
    ax[1][2].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax[1][2].xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax[1][2].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax[1][2].set_yticks([0.5, 1.0, 1.5])
    ax[1][2].set_xlabel('$\mathrm{log}_{10}$ ($k$ $\mathrm{[km^{-1}\,s])}$')
    ax[1][2].legend()

    ax[1][2].spines['top'].set_linewidth(line_width_ax)
    ax[1][2].spines['right'].set_linewidth(line_width_ax)
    ax[1][2].spines['bottom'].set_linewidth(line_width_ax)
    ax[1][2].spines['left'].set_linewidth(line_width_ax)


    if save_path != None:
        plt.savefig(save_path, bbox_inches='tight') 
    plt.subplots_adjust(right=0.95)
    plt.show()




def get_filtered_samples(samples, fixed_param, fixed_value, tolerance, irsic):
    alpha_samples = samples.getParams().alpha
    #if irsic ==False:
    fwdm_samples = samples.getParams().fwdm
    weights = samples.weights
    loglikes = samples.loglikes
    
    if fixed_param == 'alpha':
        #filtered_indices = np.abs(alpha_samples - fixed_value) < tolerance 
        #print(f"Filtered indices sum: {np.sum(filtered_indices)}")
        #if irsic==False:
        filtered_indices = np.abs(alpha_samples - fixed_value) < tolerance
        filtered_fwdm_samples = fwdm_samples[filtered_indices]
        #if irsic==True:
        #    filtered_indices = np.abs(samples.samples[:, 0] - fixed_value) < tolerance
        #    filtered_fwdm_samples = samples.samples[filtered_indices]  # Extract only the second column (fwdm)

        n_samples = int(np.shape(filtered_fwdm_samples)[0])

        print('Number of filtered samples is', np.shape(filtered_fwdm_samples))
        print(f"Filtered indices sum: {np.sum(filtered_indices)}")
        filtered_weights = weights[filtered_indices]
        filtered_loglikes = loglikes[filtered_indices]

        filtered_samples = MCSamples(
        samples=np.array([filtered_fwdm_samples]).T,  # Transpose to make it 2D
        names=['fwdm'],  labels = [r'f_{\mathrm{WDM}}'],# Parameter name
        ranges = {'fwdm': [0,1]}, weights = filtered_weights, loglikes = filtered_loglikes)# LaTeX-style label for plotting

     
    if fixed_param == 'fwdm':
        filtered_indices = np.abs(fwdm_samples - fixed_value) < tolerance 
        filtered_alpha_samples = alpha_samples[filtered_indices]

        print('Number of filtered samples is', np.shape(filtered_alpha_samples))
        n_samples = int(np.shape(filtered_alpha_samples)[0])

        filtered_weights = weights[filtered_indices]
        filtered_loglikes = loglikes[filtered_indices]

        filtered_samples = MCSamples(
        samples=np.array([filtered_alpha_samples]).T,  # Transpose to make it 2D
        names=['alpha'],  # Parameter name
        labels=[r'{m_{\mathrm{WDM}}}^{-1}\,\mathrm{(keV^{-1})}'],  ranges = {'alpha': [0,1]}, weights = filtered_weights, loglikes = filtered_loglikes)# LaTeX-style label for plotting

    return filtered_samples


def process_filtered_samples(samples, fixed_param, non_fixed_param, value, irsic, tolerance):

    """
    Filter the samples to contain fixed alpha or fwdm values plus/minus some tolerance.
    """
    filtered_samples_list = []
    n_samples_list = []
    #tolerance = 1e-4 # we have checked that smaller tolerances never gives enough samples

    
    filtered_samples = get_filtered_samples(samples, fixed_param, value, tolerance, irsic)
    #filtered_samples_list.append(filtered_samples)



    # Get 1,2,3 sigma constraints:
    sigma = 1-0.682689492137086 

    ### more accurately                                                                                                                                                                  
    two_sigma = 1-0.954499736103642                                                                    #                
    three_sigma = 1-0.997300203936740                                                                                     
    four_sigma = 1-0.999936657516334# = 4sigma                                                                                       
    five_sigma = 1-0.999999426696856 #= 5sigma

    upper_bound = (filtered_samples.confidence(non_fixed_param, limfrac=sigma, upper=True))
    upper_bound_twosig = (filtered_samples.confidence(non_fixed_param, limfrac=two_sigma, upper=True))
    upper_bound_threesig = (filtered_samples.confidence(non_fixed_param, limfrac=three_sigma, upper=True))


    if fixed_param == 'alpha':
        print(rf'Fixing mwdm = {1/value}:')# excludes fwdm > {upper_bound} at 1 sigma')
        print(rf'fwdm < {upper_bound_twosig} is excluded at 2 sigma')
        # print(rf'fwdm < {upper_bound_threesig} is excluded at 3 sigma')

    if fixed_param == 'fwdm':
        print(rf'Fixing fwdm = {value}:')# excludes alpha >', upper_bound, 'at 1 sigma')
        # print('mwdm <', 1/upper_bound, 'keV is excluded at 1 sigma')  
        print('mwdm <', 1/upper_bound_twosig, 'keV  is excluded at 2 sigma')
        # print('mwdm <', 1/upper_bound_threesig, 'keV is excluded at 3 sigma')

    return filtered_samples



def get_2sigma_filtered_samples(samples, fixed_param, non_fixed_param, value, irsic, tolerance):

    """
    Filter the samples to contain fixed alpha or fwdm values plus/minus some tolerance.
    """
    filtered_samples_list = []
    n_samples_list = []
   # tolerance = 1e-1 # we have checked that smaller tolerances never gives enough samples

    
    filtered_samples = get_filtered_samples(samples, fixed_param, value, tolerance, irsic)
    #filtered_samples_list.append(filtered_samples)



    # Get 1,2,3 sigma constraints:
    sigma = 1-0.682689492137086 

    ### more accurately                                                                                                                                                                  
    two_sigma = 1-0.954499736103642                                                                    #                
    three_sigma = 1-0.997300203936740                                                                                     
    four_sigma = 1-0.999936657516334# = 4sigma                                                                                       
    five_sigma = 1-0.999999426696856 #= 5sigma

    upper_bound = (filtered_samples.confidence(non_fixed_param, limfrac=sigma, upper=True))
    upper_bound_twosig = (filtered_samples.confidence(non_fixed_param, limfrac=two_sigma, upper=True))
    upper_bound_threesig = (filtered_samples.confidence(non_fixed_param, limfrac=three_sigma, upper=True))


    if fixed_param == 'alpha':
        print(rf'Fixing mwdm = {1/value}:')# excludes fwdm > {upper_bound} at 1 sigma')
        print(rf'fwdm < {upper_bound_twosig} is excluded at 2 sigma')
        # print(rf'fwdm < {upper_bound_threesig} is excluded at 3 sigma')

    if fixed_param == 'fwdm':
        print(rf'Fixing fwdm = {value}:')# excludes alpha >', upper_bound, 'at 1 sigma')
        # print('mwdm <', 1/upper_bound, 'keV is excluded at 1 sigma')  
        print('mwdm <', 1/upper_bound_twosig, 'keV  is excluded at 2 sigma')
        upper_bound_twosig = 1/upper_bound_twosig
        # print('mwdm <', 1/upper_bound_threesig, 'keV is excluded at 3 sigma')

    return upper_bound_twosig


# Get Irsic+24 filtered samples:
def get_irsic_filtered(samples_irsic):
    fixed_value = 0
    tolerance = 0.01
    fqso_samples = samples.getParams().fqso
    if fixed_param == 'fqso':
        filtered_indices = np.abs(fqso_samples - fixed_value) < tolerance
        filtered_fqso_samples = fqso_samples[filtered_indices]
    
        n_samples = int(np.shape(filtered_fqso_samples)[0])

        print('Number of filtered samples is', np.shape(filtered_fqso_samples))
        print(f"Filtered indices sum: {np.sum(filtered_indices)}")
        filtered_weights = weights[filtered_indices]
        filtered_loglikes = loglikes[filtered_indices]

        filtered_samples = MCSamples(
        samples=np.array([filtered_fqso_samples]).T,  # Transpose to make it 2D
        names=['alpha'],  labels=[r'{m_{\mathrm{WDM}}}^{-1}\,\mathrm{(keV^{-1})}'],# Parameter name
        ranges = {'alpha': [0,1e16]}, weights = filtered_weights, loglikes = filtered_loglikes)# LaTeX-style label for plotting

    return filtered_samples


def get_1d_fwdm(filtered_alpha_1, filtered_alpha_1_other,
                filtered_alpha_05, filtered_alpha_05_other,
                filtered_alpha_03, filtered_alpha_03_other,save_path, analysis):

    from getdist import plots

    # save_path = None
    g = plots.get_single_plotter(width_inch=16)
    fontsize = 18
    g.settings.title_limit_fontsize = fontsize
    g.settings.axes_fontsize = fontsize
    g.settings.axes_labelsize = fontsize
    g.settings.legend_fontsize = fontsize
    g.settings.fontsize = fontsize
    g.settings.legend_frame = False

    upper_kwargs = {'contour_ls': ['--']}

    g.make_figure(nplot=3, nx=3, ny=1, xstretch=9, ystretch=10)  # Increase xstretch for more horizontal space
    plt.subplots_adjust(wspace=1.0, hspace=0.8)  # Increase wspace for horizontal separation

    # Plot the data
    # Plot the data
    g.plot_1d(filtered_alpha_1, 'fwdm', title_limit=2, ax=0, colors=['r'], lims=(0, 1))
    g.plot_1d(filtered_alpha_1_other, 'fwdm', title_limit=2, ax=0, colors=['g'], lims=(0, 1))
    g.add_legend([f'{analysis}', 'Irsic+24'], legend_loc='upper right', fontsize = 15)



    # Add a combined legend
    # g.add_legend([n_alpha_1, n_alpha_1_other], colors= ['r', 'g'],legend_loc='upper right')
    g.add_text(r'${m_{\mathrm{WDM}}}$ = 1 keV', 0.97, 0.4)

    g.plot_1d(filtered_alpha_05, 'fwdm', title_limit=2, ax=1, colors=['r', 'g', 'b'], lims=(0, 1))# linestyle='--')
    g.plot_1d(filtered_alpha_05_other, 'fwdm', title_limit=2, ax=1, colors=['g', 'g', 'b'], lims=(0, 1))


    g.add_text(r'${m_{\mathrm{WDM}}}$ = 2 keV', 0.97, 0.4)

    g.plot_1d(filtered_alpha_03, 'fwdm', title_limit=2, ax=2, colors=['r', 'g', 'b'], lims=(0, 1), linestyle='--')
    g.plot_1d(filtered_alpha_03_other, 'fwdm', title_limit=2, ax=2, colors=['g', 'g', 'b'], lims=(0, 1))


    g.add_text(r'${m_{\mathrm{WDM}}}$ = 3 keV', 0.97, 0.4)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')







def get_alpha_vs_fwdm(samples_list, legends_list, colors_list, save_path, chain_irsic):#, bestfit_samples_ir, high_err_ir, twosig_high_err_ir):
    """
    Get 2d alha vs fwdm plot
    """
    from getdist import plots
    # chain_irsic = 'chain_bo_WDM_pRELICS_T0G_moC'
    samples_irsic  = get_samples_irsic(chain_irsic)
    one_sigma, two_sigma =  get_sigma_irsic_alpha(samples_irsic)
    
    g = plots.get_subplot_plotter(subplot_size=2, width_inch=6)
    print('Changes saved')
    g.settings.axes_fontsize=15
    g.settings.axes_labelsize=17
    g.settings.legend_fontsize=15
    g.settings.fontsize = 15
    g.settings.legend_frame = False
    g.settings.alpha_factor_contour_lines=0.45
    g.settings.alpha_filled_add = 0.55
    g.plot_2d(
        samples_list, 'alpha',
        'fwdm', filled=True, colors = colors_list)
    #g.add_legend(legend_list, legend_loc='upper right', fontsize = 15)
    
    ax = g.fig.gca()  # Get the matplotlib axis from the getdist plotter
    #ax.axvspan(0, one_sigma, color='dimgray', alpha=0.3)  # Add shaded region
    #ax.axvspan(0, two_sigma, color='gray', alpha=0.2)

    ax.axvline(one_sigma, linestyle='--', color='black', label='1$\sigma$ Irsic+24', ymin=0, ymax=1)
    ax.axvline(two_sigma, linestyle='--', color='dimgray', label='2$\sigma$ Irsic+24', ymin=0, ymax=1)
    #Add shaded region

    plt.legend(fontsize = 15, frameon = False,  loc=(0.4, 0.8))
    plt.ylim(0, 1.01)
    plt.xlim(0, 1.01)
    legend_elements = []
    #if 'patchy' in save_path:
    #    
    #    legend_elements = [mpatches.Patch(color='dimgray', alpha=0.3, label='1$\sigma$ Irsic+24 Patchy'),
    #                       mpatches.Patch(color='gray', alpha=0.3,label='2$\sigma$ Irsic+24 Patchy')]
    #    for legend, color in zip(legends_list, colors_list):
    #        legend_elements.append(mpatches.Patch(color=color, alpha=0.4, label=legend))

    #else: 
    #    legend_elements = [mpatches.Patch(color='dimgray', alpha=0.3, label='1$\sigma$ Irsic+24'),
    #                      mpatches.Patch(color='gray', alpha=0.3,label='2$\sigma$ Irsic+24')]
    for legend, color in zip(legends_list, colors_list):
        legend_elements.append(mpatches.Patch(color=color, alpha=0.5, label=legend))


    # Add points at fixed alpha and fwdm values
    fixed_alpha_values = [0, 1/4, 1/3, 1/2, 1]  # Example fixed alpha values
    fixed_fwdm_values = [0, 0.125, 0.25, 0.5, 1]  # Example fixed fwdm values

    for alpha_value in fixed_alpha_values:
        for fwdm_value in fixed_fwdm_values:
            ax.scatter(alpha_value, fwdm_value, color='black', marker='*', s=45)  

    # Combine legend_list and legend_elements
    handles, labels = ax.get_legend_handles_labels()
    combined_handles = legend_elements + handles

    plt.legend(handles=combined_handles, fontsize=13, loc='upper left', bbox_to_anchor=(0.3, 1.0), frameon=False)


    #plt.legend(handles=legend_elements + plt.gca().get_legend_handles_labels()[0], fontsize=15, loc='upper left', bbox_to_anchor=(0.4, 1.0), frameon=False)

    
    if save_path != None:
        plt.savefig(save_path, bbox_inches='tight') 



def get_u0_2d(samples_list, color_list, legend_list, save_path, name_chain):#, bf_irsic, high_irsic, low_irsic,  twosig_high_irsic):

    # 2D plot:
    from getdist import plots
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Get Irsic+24 best fit for a chain:
    samples_irsic = get_samples_irsic(name_chain)
    bestfit, onesigma_all = get_bf_sigma_irsic_all(name_chain, samples_irsic)
    one_sigma, two_sigma =  get_sigma_irsic_alpha(samples_irsic)
    print('One_sigma bound', one_sigma)
    print('Bestfit for alpha', bestfit[-1])
    print('Best fit for T0 at z = 4.2', bestfit[0])
    print('SIgma values', onesigma_all[0][0], onesigma_all[0][1])
    

    g = plots.get_single_plotter(width_inch=16)
    g.settings.title_limit_fontsize = 15
    g.settings.axes_fontsize=15
    g.settings.axes_labelsize=15
    g.settings.legend_fontsize=15
    g.settings.fontsize = 15
    g.settings.legend_frame = False
    g.settings.alpha_filled_add = 0.1

    g.settings.alpha_factor_contour_lines=0.45
    g.settings.alpha_filled_add = 0.4

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
    
    T0_value = np.arange(0.5, 1.5, 0.1)

    print('Limits that we would expect',onesigma_all[1][1]-bestfit[1], onesigma_all[2][1]-bestfit[2])

    # z= 4.2:
    a = 10
    y1_valuez42 = 0.6 * a * T0_value**1.7
    y2_valuez42 = 1.2 * a * T0_value**1.7
    y3_valuez42 = -5*T0_value + 7.47
    lims = [0.5, 1.4, 3.9, 20]
    print('Changes saved')
    print('Limists are', lims)
    g.settings.alpha_filled_add = 0.8
    g.plot_2d(samples_list, ['T0z42','u0z42'], lims=lims, filled=True, ax=0,  colors = color_list, alpha = 0.7)
    plt.fill_between(T0_value, y1_valuez42, y2_valuez42, color='gray', alpha=0.2)
    plt.fill_between(T0_value, y3_valuez42, min(y3_valuez42), color='white')
    plt.xlim(lims[0], lims[1])
    plt.ylim(lims[2], lims[3])
   # g.plot_2d(samples_list, ['T0z42','u0z42'], filled=True, ax=0,  colors = color_list, alpha = 0.3)
   
    plt.errorbar(
        [T0_ref[0]], [u0_ref[0]],
        xerr=[[dT0_low[0]], [dT0_high[0]]], yerr=[[du0_low[0]], [du0_high[0]]],
         fmt = '*', markersize = markersize, label='Boera+19', capsize=5, color = 'purple'
    )
    
    plt.errorbar(
        [T0_r21_ref[0]], [u0_r21_ref[0]],
        xerr=[[dT0_r21_low[0]], [dT0_r21_high[0]]], yerr=[[du0_r21_low[0]], [du0_r21_high[0]]],
         fmt = 'v', markersize = markersize, label='Rogers+21', capsize=5, color = 'tab:green'
    )
    plt.errorbar(
        [T0_v22_ref[0]], [u0_v22_ref[0]],
        xerr=[[dT0_v22_low[0]], [dT0_v22_high[0]]], yerr=[[du0_v22_low[0]], [du0_v22_low[0]]],
        fmt = '^', markersize = markersize,  label='Villasenor+22', capsize=5, color = 'tab:red'
    )
    # Irsic+24:
    plt.errorbar(
        [bestfit[0]], [bestfit[9]],
        xerr=[[bestfit[0] - onesigma_all[0][0]], [onesigma_all[0][1]-bestfit[0]]], yerr=[[bestfit[9] - onesigma_all[9][0]], [onesigma_all[9][1]-bestfit[9]]],
        fmt = 's', markersize = markersize-3,  capsize=5, color = 'saddlebrown'
    )

    # #Best-fit samples:
    # plt.scatter(bestfit_samples[0], bestfit_samples[9], marker='*', s=90, color = 'black')



    plt.ylim(np.min(u0_gridz42), np.max(u0_gridz42))
    plt.xlim(0.5, 1.4)


    # z= 4.6:
    y1_valuez46 = 0.6 * a * T0_value**1.7
    y2_valuez46 = 1.3 * a * T0_value**1.7
    y3_valuez46 = -5*T0_value + 6.85
    g.plot_2d( samples_list, ['T0z46','u0z46'], filled=True, ax=1,  colors = color_list, alpha = 0.7)
    plt.fill_between(T0_value, y1_valuez46, y2_valuez46, color='gray', alpha=0.2)
    plt.fill_between(T0_value, y3_valuez46, min(y3_valuez46), color='white')

    plt.errorbar(
        [T0_ref[1]], [u0_ref[1]],
        xerr=[[dT0_low[1]], [dT0_high[1]]], yerr=[[du0_low[1]], [du0_high[1]]],
        fmt = '*', markersize = markersize, capsize=5, color = 'purple'
    )

    plt.errorbar(
        [T0_r21_ref[1]], [u0_r21_ref[1]],
        xerr=[[dT0_r21_low[1]], [dT0_r21_high[1]]], yerr=[[du0_r21_low[1]], [du0_r21_high[1]]],
        fmt = 'v', markersize = markersize,  label='Rogers+21', capsize=5, color = 'tab:green'
    )
    plt.errorbar(
        [T0_v22_ref[1]], [u0_v22_ref[1]],
        xerr=[[dT0_v22_low[1]], [dT0_v22_high[1]]], yerr=[[du0_v22_low[1]], [du0_v22_low[1]]],
        fmt = '^', markersize = markersize,  label='Villasenor+22', capsize=5, color = 'tab:red'
    )


    #Irsic_24:
    plt.errorbar(
            [bestfit[1]], [bestfit[10]],
            xerr=[[bestfit[1] - onesigma_all[1][0]], [ onesigma_all[1][1]-bestfit[1]]], yerr=[[bestfit[10] - onesigma_all[10][0]], [onesigma_all[10][1]-bestfit[10]]],
            fmt = 's', markersize = markersize-3, capsize=5, color = 'saddlebrown'
        )
#     #Best-fit:
#     plt.scatter(bestfit_samples[1], bestfit_samples[10], marker='*', s=90, color = 'black')
#     plt.ylim(np.min(u0_gridz46), np.max(u0_gridz46))
#     plt.xlim(0.5, 1.4)



    y1_valuez5 = 0.4 * a * T0_value**1.7
    y2_valuez5 = 1.1 * a * T0_value**1.7
    y3_valuez5 = -5*T0_value + 5.87
    g.plot_2d(samples_list, ['T0z5','u0z5'], filled=True, ax=2,  colors = color_list, alpha=0.7)
    plt.fill_between(T0_value, y1_valuez5, y2_valuez5, color='gray', alpha=0.2)
    plt.fill_between(T0_value, y3_valuez5, min(y3_valuez5), color='white')
    g.plot_2d(samples_list, ['T0z5','u0z5'], filled=True, ax=2,  colors = color_list, alpha=0.3)
    
    plt.errorbar(
        [T0_ref[2]], [u0_ref[2]],
        xerr=[[dT0_low[2]], [dT0_high[2]]], yerr=[[du0_low[2]], [du0_high[2]]],
            fmt='*', markersize = markersize, capsize=5, color = 'purple', label = 'Boera+19')

    plt.errorbar(
        [T0_r21_ref[2]], [u0_r21_ref[2]],
        xerr=[[dT0_r21_low[2]], [dT0_r21_high[2]]], yerr=[[du0_r21_low[2]], [du0_r21_high[2]]],
        fmt = 'v', markersize = markersize, label='Rogers+21', capsize=5, color = 'tab:green'
    )
    plt.errorbar(
        [T0_v22_ref[2]], [u0_v22_ref[2]],
        xerr=[[dT0_v22_low[2]], [dT0_v22_high[2]]], yerr=[[du0_v22_low[2]], [du0_v22_low[2]]],
        fmt = '^', markersize = markersize,  label= 'Villasenor+22', capsize=5, color = 'tab:red')

    plt.errorbar(
            [bestfit[2]], [bestfit[11]],
            xerr=[[bestfit[2] - onesigma_all[2][0]], [ onesigma_all[2][1]-bestfit[2]]], yerr=[[bestfit[11] - onesigma_all[11][0]], [onesigma_all[11][1]-bestfit[11]]],
            fmt = 's', markersize = markersize-3,  capsize=5, color = 'saddlebrown'
        )

    # #Best-fit samples:
    # plt.scatter(bestfit_samples[2], bestfit_samples[11], marker='*', s=90, color = 'black')




    custom_lines = [
        Line2D([0], [0], marker='^',color='tab:red',  label='Villasenor+22', linestyle='None',markeredgewidth=1, markersize=11),
        
        Line2D([0], [0], marker='v', color='tab:green',  label='Rogers+21', linestyle='None', markeredgewidth=1.5,markersize=11),
        
        Line2D([0], [0], marker='*', color='purple',  label='Boera+19', linestyle='None', markeredgewidth=1.5, markersize=11),
        Line2D([0], [0], marker='s', color='saddlebrown',  label='Irsic+24', linestyle='None', markeredgewidth=1.5, markersize=markersize-3)
        # Line2D([0], [0], marker='v', color='black',  label='Best-fit $u_{0}$-$T_{0}$', linestyle='None', markeredgewidth=1.5, markersize=11)
    ]


    plt.legend(handles=custom_lines, loc='best', fontsize =15, frameon=False)
    plt.ylim(np.min(u0_gridz5), np.max(u0_gridz5))
    plt.xlim(0.5, 1.4)



    #Row 2:
    g.plot_2d(samples_list, ['alpha','u0z42'], filled=True, ax=3, colors = color_list,  alpha = 0.7) # label= {'alpha': '{m_{\mathrm{WDM}}^{-1}}'})
     #g.add_legend(legend_list, legend_loc='upper right', fontsize=15, frameon=False)

    # Irsic+24 in bands:
    ax = g.fig.gca()  # Get the matplotlib axis from the getdist plotter
    # Plot Irsic+24 as best-fit and 1sigma:
    # plt.errorbar(
    #     [bestfit[-1]], [bestfit[9]],
    #     xerr=[[0], [bestfit[-1]+one_sigma]], yerr=[[bestfit[9] - onesigma_all[9][0]], [onesigma_all[9][1]-bestfit[9]]],
    #     fmt = 's', markersize = markersize-2,  capsize=5, color = 'saddlebrown'
    # )

    # Plot Irsic+24 shifted to the right from 0 and the 2 sigma limit:
    # two_sigma
    # plt.errorbar(
    #     [0.001], [bestfit[9]],
    #     xerr=[[0], [0.001+tw0_sigma]], yerr=[[bestfit[9] - onesigma_all[9][0]], [onesigma_all[9][1]-bestfit[9]]],
    #     fmt = 's', markersize = markersize-2,  capsize=5, color = 'saddlebrown'
    # )
    plt.errorbar(
    [0.015], [bestfit[9]],
    yerr=[[bestfit[9] - onesigma_all[9][0]], [onesigma_all[9][1]-bestfit[9]]],
    fmt='s', markersize=markersize-3, capsize=5, color='saddlebrown')

    # Add an arrow for the x error bar
    plt.annotate(
        '', xy=(0.015 + two_sigma, bestfit[9]), xytext=(0.015, bestfit[9]),
        arrowprops=dict(arrowstyle='->', color='saddlebrown')
    )




    # ax.axvspan(0, high_irsic['alpha'], color='darkblue', alpha=0.3)  # Add shaded region
    # ax.axvspan(0, twosig_high_irsic['alpha'], color='b', alpha=0.2)


    # #Best-fit samples:
    # plt.scatter(bestfit_samples[-2], bestfit_samples[9], marker='*', s=90, color = 'black')


   
    # legend_elements = [
    # mpatches.Patch(color='darkblue', alpha=0.3, label='1$\sigma$ Irsic+24'),
    # mpatches.Patch(color='b', alpha=0.3,label='2$\sigma$ Irsic+24')]
    # plt.legend(handles=legend_elements + plt.gca().get_legend_handles_labels()[0], fontsize=15, loc='upper left', bbox_to_anchor=(0.4, 1.0), frameon=False)
    legend_elements = []
    for legend, color in zip(legend_list, color_list):
        legend_elements.append(mpatches.Patch(color=color, alpha=0.7, label=legend))

     # Combine legend_list and legend_elements
    handles, labels = ax.get_legend_handles_labels()
    combined_handles = legend_elements + handles

    plt.legend(handles=combined_handles, fontsize=13, loc='upper left', bbox_to_anchor=(0.01, 1.0), frameon=False)


    g.plot_2d(samples_list, ['alpha','u0z46'], filled=True, ax=4, colors = color_list, alpha=0.7)
    
    # Irsic+24 in bands:

    # plt.errorbar(
    #     [bestfit[-1]], [bestfit[10]],
    #     xerr=[[0], [bestfit[-1]+one_sigma]], yerr=[[bestfit[10] - onesigma_all[10][0]], [onesigma_all[10][1]-bestfit[10]]],
    #     fmt = 's', markersize = markersize-2,  capsize=5, color = 'saddlebrown'
    # )

    plt.errorbar(
    [0.015], [bestfit[10]],
    yerr=[[bestfit[10] - onesigma_all[10][0]], [onesigma_all[10][1]-bestfit[10]]],
    fmt='s', markersize=markersize-3, capsize=5, color='saddlebrown')

    # Add an arrow for the x error bar
    plt.annotate(
        '', xy=(0.015 + two_sigma, bestfit[10]), xytext=(0.015, bestfit[10]),
        arrowprops=dict(arrowstyle='->', color='saddlebrown')
    )
    
    ax = g.fig.gca()  # Get the matplotlib axis from the getdist plotter
    # ax.axvspan(0, high_irsic['alpha'], color='darkblue', alpha=0.3)  # Add shaded region
    # ax.axvspan(0, twosig_high_irsic['alpha'], color='b', alpha=0.2)


    #Best-fit samples:
    # plt.scatter(bestfit_samples[-2], bestfit_samples[10], marker='*', s=90, color = 'black')

    g.plot_2d(samples_list, ['alpha','u0z5'], filled=True, ax=5, colors = color_list, alpha=0.7)

    # Irsic+24 in bands:
    # plt.errorbar(
    #     [bestfit[-1]], [bestfit[11]],
    #     xerr=[[0], [bestfit[-1]+one_sigma]], yerr=[[bestfit[11] - onesigma_all[11][0]], [onesigma_all[11][1]-bestfit[11]]],
    #     fmt = 's', markersize = markersize-2, capsize=5, color = 'saddlebrown'
    # )

    plt.errorbar(
    [0.015], [bestfit[11]],
    yerr=[[bestfit[11] - onesigma_all[11][0]], [onesigma_all[11][1]-bestfit[11]]],
    fmt='s', markersize=markersize-3, capsize=5, color='saddlebrown')

    # Add an arrow for the x error bar
    plt.annotate(
        '', xy=(0.015 + two_sigma, bestfit[11]), xytext=(0.015, bestfit[11]),
        arrowprops=dict(arrowstyle='->', color='saddlebrown')
    )
    
    ax = g.fig.gca()  # Get the matplotlib axis from the getdist plotter
    # ax.axvspan(0, high_irsic['alpha'], color='darkblue', alpha=0.3)  # Add shaded region
    # ax.axvspan(0, twosig_high_irsic['alpha'], color='b', alpha=0.2)


    # #Best-fit samples:
    # plt.scatter(bestfit_samples[-2], bestfit_samples[11], marker='*', s=90, color = 'black')

    #Row 3:
    g.plot_2d(samples_list, ['fwdm','u0z42'], filled=True, ax=6, colors = color_list, alpha=0.7)
    
    #Best-fit samples:
    # plt.scatter(bestfit_samples[-1], bestfit_samples[9], marker='*', s=90, color = 'black')

    g.plot_2d(samples_list, ['fwdm','u0z46'], filled=True, ax=7, colors = color_list, alpha=0.7)

    # #Best-fit samples:
    # plt.scatter(bestfit_samples[-1], bestfit_samples[10], marker='*', s=90, color = 'black')

    g.plot_2d(samples_list, ['fwdm','u0z5'], filled=True, ax=8, colors = color_list, alpha = 0.7)
    
    #Best-fit samples:
    # plt.scatter(bestfit_samples[-1], bestfit_samples[11], marker='*', s=90, color = 'black')
    print('One sigma for T0 z = 4.6', onesigma_all[1][1])
    print('One sigma for T0 z = 5.0', onesigma_all[2][1])
    if save_path != None:
        plt.savefig(save_path, bbox_inches='tight') 


#For other analysis where we do not want to show all the redshift information:
def get_plots_other(samples_list, color_list, legend_list, save_path, name_chain):
      # 2D plot:
    from getdist import plots
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Get Irsic+24 best fit for a chain:
    samples_irsic = get_samples_irsic(name_chain)
    bestfit, onesigma_all = get_bf_sigma_irsic_all(name_chain, samples_irsic)
    one_sigma, two_sigma =  get_sigma_irsic_alpha(samples_irsic)
    print('One_sigma bound', one_sigma)
    print('Bestfit for alpha', bestfit[-1])
    print('Best fit for T0 at z = 4.2', bestfit[0])
    print('SIgma values', onesigma_all[0][0], onesigma_all[0][1])
    

    g = plots.get_single_plotter(width_inch=16)
    g.settings.title_limit_fontsize = 20
    g.settings.axes_fontsize=20
    g.settings.axes_labelsize=20
    g.settings.legend_fontsize=20
    g.settings.fontsize = 20
    g.settings.legend_frame = False
    g.settings.alpha_filled_add = 0.1
    g.settings.alpha_factor_contour_lines=0.4
    g.settings.alpha_filled_add = 0.7
    g.make_figure(nplot=4, nx=2, ny=2, xstretch=7, ystretch = 9)
    markersize = 11

    plt.subplots_adjust(wspace=4, top=0.95, bottom=0.05)
    # Settings
    g.settings.constrained_layout = True

    # # Set space between each subplot
    g.fig.subplots_adjust(hspace = 0.2, wspace = 0.2)

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
    
    T0_value = np.arange(0.5, 1.5, 0.1)

    print('Limits that we would expect',onesigma_all[1][1]-bestfit[1], onesigma_all[2][1]-bestfit[2])

    # z= 4.2:
    a = 10
    y1_valuez42 = 0.6 * a * T0_value**1.7
    y2_valuez42 = 1.2 * a * T0_value**1.7
    y3_valuez42 = -5*T0_value + 7.47
    lims = [0.5, 1.4, 3.9, 20]
    print('Changes saved')
    print('Limists are', lims)

    g.plot_2d(samples_list, ['T0z42','u0z42'], lims=lims, filled=True, ax=1,  colors = color_list)
    plt.fill_between(T0_value, y1_valuez42, y2_valuez42, color='gray', alpha=0.2)
    plt.fill_between(T0_value, y3_valuez42, min(y3_valuez42), color='white')
    plt.xlim(lims[0], lims[1])
    plt.ylim(lims[2], lims[3])
   # g.plot_2d(samples_list, ['T0z42','u0z42'], filled=True, ax=0,  colors = color_list, alpha = 0.3)
   
    plt.errorbar(
        [T0_ref[0]], [u0_ref[0]],
        xerr=[[dT0_low[0]], [dT0_high[0]]], yerr=[[du0_low[0]], [du0_high[0]]],
         fmt = '*', markersize = markersize, label='Boera+19', capsize=5, color = 'purple'
    )
    
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
    # Irsic+24:
    plt.errorbar(
        [bestfit[0]], [bestfit[9]],
        xerr=[[bestfit[0] - onesigma_all[0][0]], [onesigma_all[0][1]-bestfit[0]]], yerr=[[bestfit[9] - onesigma_all[9][0]], [onesigma_all[9][1]-bestfit[9]]],
        fmt = '*', markersize = markersize,  capsize=5, color = 'b'
    )

    custom_lines = [
        Line2D([0], [0], marker='*',color='tab:red',  label='Villasenor+22', linestyle='None',markeredgewidth=1, markersize=11),
        
        Line2D([0], [0], marker='*', color='tab:green',  label='Rogers+21', linestyle='None', markeredgewidth=1.5,markersize=11),
        
        Line2D([0], [0], marker='*', color='purple',  label='Boera+19', linestyle='None', markeredgewidth=1.5, markersize=11),
        Line2D([0], [0], marker='*', color='b',  label='Irsic+24', linestyle='None', markeredgewidth=1.5, markersize=11)
    ]

    plt.legend(handles=custom_lines, loc='best', frameon=False, fontsize= 20)
    plt.ylim(3.9, 20)
    plt.xlim(0.5, 1.4)

        #Row 2:
    g.plot_2d(samples_list, ['alpha','u0z42'], filled=True, ax=2, colors = color_list) # label= {'alpha': '{m_{\mathrm{WDM}}^{-1}}'})
     #g.add_legend(legend_list, legend_loc='upper right', fontsize=15, frameon=False)

    # Irsic+24 in bands:
    ax = g.fig.gca()  # Get the matplotlib axis from the getdist plotter
    plt.errorbar(
        [bestfit[-1]], [bestfit[9]],
        xerr=[[0], [bestfit[-1]+one_sigma]], yerr=[[bestfit[9] - onesigma_all[9][0]], [onesigma_all[9][1]-bestfit[9]]],
        fmt = '*', markersize = markersize,  capsize=5, color = 'b'
    )

    legend_elements = []
    for legend, color in zip(legend_list, color_list):
        legend_elements.append(mpatches.Patch(color=color, alpha=0.4, label=legend))

     # Combine legend_list and legend_elements
    handles, labels = ax.get_legend_handles_labels()
    combined_handles = legend_elements + handles

    plt.legend(handles=combined_handles, loc='upper left', bbox_to_anchor=(0.01, 1.0), frameon=False, fontsize = 20)

    g.plot_2d(samples_list, ['fwdm','u0z42'], filled=True, ax=3, colors = color_list)
    g.plot_2d(samples_list, ['alpha','fwdm'], filled=True, ax=0, colors = color_list)
        
    ax = g.fig.gca()  # Get the matplotlib axis from the getdist plotter
    ax.axvspan(0, one_sigma, color='dimgray', alpha=0.3)  # Add shaded region
    ax.axvspan(0, two_sigma, color='gray', alpha=0.2)

    plt.legend(fontsize = 15, frameon = False)
    if 'patchy' in save_path:
        
        legend_elements = [mpatches.Patch(color='dimgray', alpha=0.3, label='1$\sigma$ Irsic+24 Patchy'),
                           mpatches.Patch(color='gray', alpha=0.3,label='2$\sigma$ Irsic+24 Patchy')]
        #for legend, color in zip(legend_list, color_list):
        #    legend_elements.append(mpatches.Patch(color=color, alpha=0.4, label=legend))

    else: 
        legend_elements = [mpatches.Patch(color='dimgray', alpha=0.3, label='1$\sigma$ Irsic+24'),
                           mpatches.Patch(color='gray', alpha=0.3,label='2$\sigma$ Irsic+24')]
        #for legend, color in zip(legend_list, color_list):
        #    legend_elements.append(mpatches.Patch(color=color, alpha=0.4, label=legend))


    # Add points at fixed alpha and fwdm values
    fixed_alpha_values = [0, 1/4, 1/3, 1/2, 1]  # Example fixed alpha values
    fixed_fwdm_values = [0, 0.125, 0.25, 0.5, 1]  # Example fixed fwdm values

    for alpha_value in fixed_alpha_values:
        for fwdm_value in fixed_fwdm_values:
            ax.scatter(alpha_value, fwdm_value, color='black', marker='*', s=40)  

    # Combine legend_list and legend_elements
    handles, labels = ax.get_legend_handles_labels()
    combined_handles = legend_elements + handles

    plt.legend(handles=combined_handles, fontsize=20, loc='upper left', bbox_to_anchor=(0.3, 1.0), frameon=False)

    if save_path != None:
        plt.savefig(save_path, bbox_inches='tight') 




        
def get_chi2_at_point(name_chains,samples, base_path, post, irsic, kmax, new_alpha):
    
    if irsic == True:
        # arr = [0.83, 0.82, 0.94, 1.13, 1.06, 1.01, 1.11, 1.43, 1.59, 6.21, 5.95, 3.49, 0.05, 1]
        bestfit, onesigma_all = get_bf_sigma_irsic_all(name_chain, samples)
        arr = bestfit
        print('Best-fit samples', arr)
        
    elif irsic == False:
        # Get best-fit parameters for samples considered:
        like_stats = samples.getLikeStats()
        n_pars = 14
        arr = [like_stats.names[i].bestfit_sample for i in range(n_pars)]
        print('Best-fit samples', arr)
    
    min_in42, max_in42, models42 = get_models_z(4.2, base_path)
    min_in46, max_in46, models46 = get_models_z(4.6, base_path)
    min_in5, max_in5, models5 = get_models_z(5.0, base_path)

    log_k, P_kRC42, P_kRC46, P_kRC5, boera_covariance_matrix = get_data()

    # Depening on interpolation method, load necessary variables:
    # Define column names
    if post ==True:
        column_names = ['Model', 'zr', 'tau_e', 'u0_4.2<=z<=12.0', 'u0_4.6<=z<=13.0', 'u0_6.0<=z<=13.0']

        # Load data from text file, skipping the first 5 rows
        u0all = pd.read_csv('/data/og313b/lyalpha/taue_u0_table.txt', skiprows=4, delim_whitespace=True, names=column_names)
        models_cosmo = ['planck1_20_1024', 'planck1_20_1024_cold', 'planck1_20_1024_hot', 'planck1_20_1024_zr525_cold', 'planck1_20_1024_zr525', 'planck1_20_1024_zr675', 'planck1_20_1024_zr750']

        # Reduce table to models for which we have u0 in wdm cosmlogies:
        filtered_u0all = u0all[u0all['Model'].isin(models_cosmo)].sort_values('u0_4.2<=z<=12.0').reset_index(drop=True)

        # Models in increasing u0 values for each redshift:
        u042 = filtered_u0all.iloc[:,3].reset_index(drop=True)
        u046 = filtered_u0all.iloc[:,4].reset_index(drop=True)
        u05 = filtered_u0all.iloc[:,5].reset_index(drop=True)


        Rs10_42 = np.load('/data/og313/lyalpha/code_wdm/analysis_RC_cdm_u0_post/Rs10_42.npy')
        Rs10_46 = np.load('/data/og313/lyalpha/code_wdm/analysis_RC_cdm_u0_post/Rs10_46.npy')
        Rs10_5 = np.load('/data/og313/lyalpha/code_wdm/analysis_RC_cdm_u0_post/Rs10_5.npy')

        print(np.shape(u042), np.shape(Rs10_42))

    if kmax ==True:

        #kmax cut analysis: re-define covariance matrix:                                             
        block1 = boera_covariance_matrix[:16, :16]
        block2 = boera_covariance_matrix[16:32, 16:32]
        block3 = boera_covariance_matrix[32:48, 32:48]

        # Remove the last three rows and columns from each (16,16) block to get (13,13) blocks       
        block1_reduced = block1[:-3, :-3]
        block2_reduced = block2[:-3, :-3]
        block3_reduced = block3[:-3, :-3]

        # Concatenate the resulting (13,13) blocks to form the final (39,39) covariance matrix       
        covariance_matrix = np.block([
            [block1_reduced, np.zeros((13, 13)), np.zeros((13, 13))],
            [np.zeros((13, 13)), block2_reduced, np.zeros((13, 13))],
            [np.zeros((13, 13)), np.zeros((13, 13)), block3_reduced]
        ])

        print('Shape of the covariance matrix', np.shape(covariance_matrix))

        # shape = boera_covariance_matrix.shape
        # updated_matrix = np.zeros(shape)
        # indices = [-3, -2, -1]

        # # Function to update only the diagonal elements in a given block
        # def update_diagonal(matrix, block_start, block_end):
        #     for i in indices:
        #         original_value = boera_covariance_matrix[block_end + i, block_end + i]
        #         updated_value = original_value * 0.50  # Get 50% of the sigma^2 in last three bins
        #         matrix[block_end + i, block_end + i] = updated_value

        # # Apply the function to the three blocks
        # update_diagonal(updated_matrix, 0, 16)   # Block1
        # update_diagonal(updated_matrix, 16, 32)  # Block2
        # update_diagonal(updated_matrix, 32, 48)  # Block3

        # covariance_matrix = updated_matrix + boera_covariance_matrix
        # print(np.diag(covariance_matrix)[13:16], np.diag(covariance_matrix)[29:32], np.diag(covariance_matrix)[45:48])

    elif kmax == False:
        covariance_matrix =boera_covariance_matrix


    print('SHAPE OF COVARIANCE MATRIX USED', np.shape(covariance_matrix))
    if irsic == True:
        bf_input_z42 = [arr[0], arr[3], arr[6], arr[9], arr[-1], 1] #Pass emulator with fwdm = 1
        print('Best-fit values used at z = 4.2',  bf_input_z42 )
    else:
        bf_input_z42 = [arr[0], arr[3], arr[6], arr[9], new_alpha, arr[-1]]
    print(bf_input_z42 )
    # Choose interpolation method:
    if post ==True:
        bf_pred_42 = get_nn_prediction_post_rs_mcmc(4.2, bf_input_z42 , min_in42, max_in42, models42, u042, u046, u05, Rs10_42, Rs10_46, Rs10_5)
    elif post==False:
        bf_pred_42 = get_nn_prediction_mcmc(4.2, bf_input_z42, min_in42, max_in42, models42, rc=True)
        # print('best fit predicted power spectra', bf_pred_42 )

    bf_pred_42 = np.array(bf_pred_42).flatten()
    if kmax == True:
        residuals = bf_pred_42[:-3] - P_kRC42[:-3]
        cov_matrix = covariance_matrix[:13, :13]
        print('Shape of residuals', residuals)
    else:
        residuals = bf_pred_42 - P_kRC42
        cov_matrix = covariance_matrix[:16, :16]

    chi42_value = compute_chi_squared(residuals.reshape(-1,1), cov_matrix)
    print(f"Chi-squared: {chi42_value}")

    
    if irsic == True:
        bf_input_z46 = [arr[1], arr[4], arr[7], arr[10], arr[-1], 1]
        print('Best-fit values used at z = 4.6',  bf_input_z46)
    else: 
        bf_input_z46 = [arr[1], arr[4], arr[7], arr[10], new_alpha, arr[-1]]
    if post == True:
        bf_pred_46 = get_nn_prediction_post_rs_mcmc(4.6, bf_input_z46, min_in46, max_in46, models46, u042, u046, u05, Rs10_42, Rs10_46, Rs10_5)
    elif post==False:
        bf_pred_46 = get_nn_prediction_mcmc(4.6, bf_input_z46, min_in46, max_in46, models46, rc=True)
    
    bf_pred_46 = np.array(bf_pred_46).flatten()
    
    if kmax == True:
        residuals = bf_pred_46[:-3] - P_kRC46[:-3]
        #print('Shape of residuals', residuals)
        cov_matrix = covariance_matrix[13:26, 13:26]
    else:
        residuals = bf_pred_46 - P_kRC46
        cov_matrix = covariance_matrix[16:32, 16:32]

    chi46_value = compute_chi_squared(residuals.reshape(-1,1), cov_matrix)

    print(f"Chi-squared: {chi46_value}")

    if irsic == True:
        bf_input_z5 = [arr[2], arr[5], arr[8], arr[11], arr[-1], 0]
        print('Best-fit values used at z = 5',  bf_input_z5 )
    else: 
        bf_input_z5 = [arr[2], arr[5], arr[8], arr[11], new_alpha, arr[-1]]
    if post==True:
        bf_pred_5 = get_nn_prediction_post_rs_mcmc(5.0, bf_input_z5, min_in5, max_in5, models5, u042, u046, u05, Rs10_42, Rs10_46, Rs10_5)
    elif post==False:
        bf_pred_5 = get_nn_prediction_mcmc(5.0, bf_input_z5, min_in5, max_in5, models5, rc=True)
    bf_pred_5 = np.array(bf_pred_5).flatten()
    if kmax == True:
        residuals = bf_pred_5[:-3] - P_kRC5[:-3]
        cov_matrix = covariance_matrix[26:, 26:]
    else:
        residuals = bf_pred_5 - P_kRC5
        cov_matrix = covariance_matrix[32:, 32:]

    chi5_value = compute_chi_squared(residuals.reshape(-1,1), cov_matrix)

    print(f"Chi-squared: {chi5_value}")

    bf_pred_5 = np.array(bf_pred_5).flatten()
    bf_pred_46 = np.array(bf_pred_46).flatten()
    bf_pred_42 = np.array(bf_pred_42).flatten()


    # FOR TABLE:
    print('BEST FIT VALUES TO PUT IN TABLE', bf_input_z46[0], bf_input_z46[3],1/ bf_input_z46[-2], bf_input_z46[-1])
    comb_chi = chi42_value + chi46_value + chi5_value
    print('COMBINED chi squared is', comb_chi)
    

    return chi42_value, chi46_value, chi5_value, bf_pred_42.flatten(), bf_pred_46.flatten(), bf_pred_5.flatten()
