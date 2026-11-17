
import sys
sys.path.append('/data/og313/lyalpha/code_wdm/nn_emulator_fwdm/')
from model_loader import get_mixed_models
from predictor import get_nn_prediction_mcmc
# from funcs_mcmc import get_prior_z, get_prior_alpha, get_prior_fwdm

import numpy as np
import os

from cobaya.model import get_model
from cobaya import run

import sys
sys.path.append('/data/og313/lyalpha/code_wdm/mcmc')
from get_boera_data import get_data
import time

import argparse
#parser = argparse.ArgumentParser(description='Process some integers.')
#parser.add_argument('--output_dir', type=str, help='Output directory for the job')
#args = parser.parse_args()

#output_dir = args.output_dir
#os.makedirs(output_dir, exist_ok=True)
#if not os.path.exists(output_dir):
#    os.makedirs(output_dir)


'Script to run cobaya for cdm_wdm for all redshifts'

current_directory = os.getcwd()
in_file42 = f"inputs4.2{current_directory.split('/')[-1].split('analysis')[1]}.txt"
out_file42 = f"ps4.2{current_directory.split('/')[-1].split('analysis')[1]}.txt"

print(in_file42, out_file42)
in_file46 = f"inputs4.6{current_directory.split('/')[-1].split('analysis')[1]}.txt"
out_file46 = f"ps4.6{current_directory.split('/')[-1].split('analysis')[1]}.txt"

in_file5 = f"inputs5.0{current_directory.split('/')[-1].split('analysis')[1]}.txt"
out_file5 = f"ps5.0{current_directory.split('/')[-1].split('analysis')[1]}.txt"

models42, min_in42, max_in42, taueff_boera_z42 = get_mixed_models(4.2, current_directory, in_file42, out_file42)
models46, min_in46, max_in46, taueff_boera_z46= get_mixed_models(4.6, current_directory, in_file46, out_file46)
models5, min_in5, max_in5, taueff_boera_z5 = get_mixed_models(5.0, current_directory, in_file5, out_file5)

log_k = np.arange(-2.2, -0.7, 0.1)
log_k, P_kRC42, P_kRC46, P_kRC5, covariance_matrix = get_data() 


tau_T0 = 0.2
tau_gamma = 0.2
tau_u0 = 5
tau_alpha = 0.1
tau_tau_effz42 = taueff_boera_z42
tau_tau_effz46 = taueff_boera_z46
tau_tau_effz5 = taueff_boera_z5

#TEST:
tau_fwdm = 0.1

T0_grid = np.arange(0.5, 1.5, 0.1)
gamma_grid = np.arange(0.9, 1.9, 0.1)
taueff_gridz42 = np.arange(0.3, 1.8, 0.1)*taueff_boera_z42
taueff_gridz46 = np.arange(0.3, 1.8, 0.1)*taueff_boera_z46
taueff_gridz5 = np.arange(0.3, 1.8, 0.1)*taueff_boera_z5
alpha_grid = [0, 1/4, 1/3, 1/2]

u0_gridz42 = [4.0239696, 4.6623777, 5.6461662, 6.5922353, 7.1101564, 8.1477417, 9.9236799, 11.614156, 13.262641, 15.009738, 18.176741, 21.106591]
u0_gridz46 = [3.6469614, 4.3163684, 5.364912, 6.3899003, 6.589134, 7.687947, 9.587889, 11.434539, 12.536159, 14.414026, 17.83328, 21.078213]
u0_gridz5 = [2.4634363, 3.4376965, 4.3868741, 4.7135148, 5.3831631, 6.3834273, 8.1135775, 9.2448624, 9.917967, 12.360053, 15.563067, 18.783147]
   

diagonal_elements = [tau_T0**2, tau_T0**2, tau_T0**2, tau_gamma**2, tau_gamma**2, tau_gamma**2, (0.104344)**2, ( 0.140348)**2,  (0.184952)**2, tau_u0**2, tau_u0**2,
                    tau_u0**2,tau_alpha**2, tau_fwdm**2  ]

jump_cov = np.diag(diagonal_elements)

#output_dir = 'T0G_1/mcmc'
def get_log_likelihood(_self=None, **params_values):
    """
    Modified likelihood function to handle parameters correctly
    """

    
    # Get parameters from _self if available
    if _self is not None and hasattr(_self, 'provider'):
        print("Getting parameters from provider")
        params = {}
        for param in _self.input_params:
            params[param] = _self.provider.get_param(param)
    else:
        print("Using provided params_values")
        params = params_values
        
    print("Final parameters being used:", params)

    # Verify we have all required parameters
    required_params = [
        'T0z42', 'gammaz42', 'taueffz42', 'u0z42',
        'T0z46', 'gammaz46', 'taueffz46', 'u0z46',
        'T0z5', 'gammaz5', 'taueffz5', 'u0z5', 
        'alpha', 'fwdm'
    ]
    
    missing_params = [p for p in required_params if p not in params]
    if missing_params:
        print(f"Missing parameters: {missing_params}")
        return -np.inf

    # Group parameters by redshift
    params_by_redshift = [
        [params['T0z42'], params['gammaz42'], params['taueffz42'], 
            params['u0z42'], params['alpha'], params['fwdm']],
        [params['T0z46'], params['gammaz46'], params['taueffz46'], 
            params['u0z46'], params['alpha'], params['fwdm']],
        [params['T0z5'], params['gammaz5'], params['taueffz5'], 
            params['u0z5'], params['alpha'], params['fwdm']]
    ]

    residuals = []
    log_prior_values = []
    redshifts = [4.2, 4.6, 5.0]
    
    for i, z in enumerate(redshifts):
        current_params = params_by_redshift[i]
        # print(f"Processing redshift {z} with parameters:", current_params)
        
        # Calculate prior
        #log_prior_value = np.log(get_prior_z(
            #   current_params[:-2],
            #  np.arange(0.5, 1.5, 0.1),
            #   np.arange(0.9, 1.9, 0.1),
            #   [taueff_gridz42, taueff_gridz46, taueff_gridz5][i],
            #   [u0_gridz42, u0_gridz46, u0_gridz5][i],
            #   [0.91555, 0.89865, 0.92865][i]
        #))
        #log_prior_values.append(log_prior_value)

        #print(np.shape(current_params))
        #  tau_effs = [taueff_boera_z42, taueff_boera_z46, taueff_boera_z5]

        prediction = get_nn_prediction_mcmc(
            z,
            current_params,
            [min_in42, min_in46, min_in5][i],
            [max_in42, max_in46, max_in5][i],
            [models42, models46, models5][i],
            rc=True
        )
        # print('Prediction', np.shape(prediction))
        residuals.append(prediction - [P_kRC42, P_kRC46, P_kRC5][i])

    # Add alpha prior
    #log_prior_alpha = np.log(get_prior_alpha(params['alpha']))

    # Add fwdm prior:
    #log_prior_fwdm = np.log(get_prior_fwdm(params['fwdm']))

    #log_prior_values.append(log_prior_alpha)
    #log_prior_values.append(log_prior_fwdm)

    # Calculate likelihood
    print('Shape of residuals', np.shape(np.array(residuals)))
    residuals = np.array(residuals).flatten()
    cov_matrix_inv = np.linalg.inv(covariance_matrix)

    log_likelihood = -0.5 * np.dot(residuals.T, np.dot(cov_matrix_inv, residuals))
    log_likelihood -= 0.5 * np.log(np.linalg.det(covariance_matrix))
    log_likelihood -= (np.shape(covariance_matrix)[0]) * 0.5 * np.log(2 * np.pi)

    #log_posterior = log_likelihood + sum(log_prior_values)

    print("Computed log-likelihood:", log_likelihood)
    return float(log_likelihood)



def log_u0_T0_z42(T0_value, u0z):
    a = 10
    y1_value = 0.6 * a * T0_value**1.7
    y2_value = 1.2 * a * T0_value**1.7
    if u0z <= y1_value or u0z >= y2_value:
        return np.log(1e-300)
    return np.log(1)
    
def log_u0_T0_z46(T0_value, u0z):
    a = 10
    y1_value = 0.6 * a * T0_value**1.7
    y2_value = 1.3 * a * T0_value**1.7
    if u0z <= y1_value or u0z >= y2_value:
        return np.log(1e-300)
    return np.log(1)
    
def log_u0_T0_z5(T0_value, u0z):
    a = 10
    y1_value = 0.4 * a * T0_value**1.7
    y2_value = 1.1 * a * T0_value**1.7
    if u0z <= y1_value or u0z >= y2_value:
        return np.log(1e-300)
    return np.log(1)


info = {
    'likelihood': {
        'experiment': {
            'external': get_log_likelihood,
            'input_params': [
                'T0z42', 'gammaz42', 'taueffz42', 'u0z42',
                'T0z46', 'gammaz46', 'taueffz46', 'u0z46',
                'T0z5', 'gammaz5', 'taueffz5', 'u0z5', 
                'alpha', 'fwdm'
            ],
            'stop_at_error': True
        }
    },
'params': {
    'T0z42': {
        'prior': {'dist': 'norm', 'loc': 0.91555, 'scale': 0.1}, 
        'ref': 0.91555,
        'latex': r'{T_{0}}^{4.2} / {10^{4}} \, \mathrm{(K)}'
    },
    'T0z46': {
        'prior': {'dist': 'norm', 'loc': 0.89865, 'scale': 0.1}, 
        'ref': 0.89865,
        'latex': r'{T_{0}}^{4.6} / {10^{4}} \, \mathrm{(K)}'
    },
    'T0z5': {
        'prior': {'dist': 'norm', 'loc': 0.92865, 'scale': 0.1}, 
        'ref': 0.92865,
        'latex': r'{T_{0}}^{5.0} / {10^{4}} \, \mathrm{(K)}'
    },
    'gammaz42': {
        'prior': {'min': 0.9, 'max': 1.8}, 
        'ref': 1.13,
        'latex': r'{\gamma}^{\,4.2}'
    },
    'gammaz46': {
        'prior': {'min': 0.9, 'max': 1.8}, 
        'ref': 1.06,
        'latex': r'{\gamma}^{\,4.6}'
    },
    'gammaz5': {
        'prior': {'min': 0.9, 'max': 1.8}, 
        'ref': 1.01,
        'latex': r'{\gamma}^{\,5.0}'
    },
    'taueffz42': {
        'prior': {'min': 0.31303136056832276, 'max': 1.7738443765538294}, 
        'ref': 1.11,
        'latex': r'{\tau_{\mathrm{eff}}}^{\,4.2}'
    },
    'taueffz46': {
        'prior': {'min': 0.42104312690706497, 'max': 2.385911052473369}, 
        'ref': 1.43,
        'latex': r'{\tau_{\mathrm{eff}}}^{\,4.6}'
    },
    'taueffz5': {
        'prior': {'min': 0.554854964068882, 'max': 3.1441781297236653}, 
        'ref': 1.59,
        'latex': r'{\tau_{\mathrm{eff}}}^{\,5.0}'
    },
    'u0z42': {
        'prior': {'min': 4.0239696, 'max': 21.106591}, 
        'ref': 6.21,
        'latex': r'u_{0}^{\,4.2}\,\mathrm{(eV\,m_{p}^{-1})}'
    },
    'u0z46': {
        'prior': {'min': 3.6469614, 'max': 21.078213}, 
        'ref': 5.95,
        'latex': r'u_{0}^{\,4.6}\,\mathrm{(eV\,m_{p}^{-1})}'
    },
    'u0z5': {
        'prior': {'min': 2.4634363, 'max': 18.783147}, 
        'ref': 3.49,
        'latex': r'u_{0}^{\,5.0}\,\mathrm{(eV\,m_{p}^{-1})}'
    },
    'alpha': {
        'prior': {'min': 0, 'max': 1e30}, 
        'ref': 0,
        'latex': r'm_{\mathrm{WDM}}'
    },
    'fwdm': {
        'prior': {'min': 0, 'max': 1}, 
        'ref': 0,
        'latex': r'f_{\mathrm{WDM}}'
    }
},

    'output': 'cobaya_chains/T0G/mcmc', 
    'sampler': {
        'mcmc': {
            "covmat": jump_cov,
                "covmat_params": [
                'T0z42', 'T0z46', 'T0z5', 
                'gammaz42', 'gammaz46', 'gammaz5',
                'taueffz42', 'taueffz46', 'taueffz5',
                'u0z42', 'u0z46', 'u0z5',
                'alpha', 'fwdm'
            ],
            "seed": 45,
            "learn_proposal": True,
            "Rminus1_stop": 0.01,
            "Rminus1_cl_stop": 0.05,
            "oversample_power": 0.5,
            'max_samples':np.inf,
            "max_tries": np.inf,
            'output_every': 60}},
    'resume': 'auto'
    }
    # 'minimizer': {
    #     'method': 'BFGS',  # Example minimizer method
    #     'tol': 1e-6,       # Example tolerance
    #     'options': {
    #         'maxiter': 1000  # Example maximum iterations
    #     }}

t1 = time.time()
updated_info, sampler = run(info)
t2 = time.time()
print(t2-t1)
