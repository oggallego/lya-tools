#%%
import sys
sys.path.append('/data/og313/iso/nn_emulator_iso/')
from model_loader import get_mixed_models
from predictor import get_nn_prediction_mcmc # does not change for 4 input model
# from funcs_mcmc import get_prior_z, get_prior_alpha, get_prior_fwdm

import numpy as np
import os

from cobaya.model import get_model
from cobaya import run

import sys
sys.path.append('/data/og313/lyalpha/code_wdm/mcmc')
from get_boera_data import get_data
import time
from scipy.optimize import curve_fit

#from scipy.interpolate import griddata
#parser = argparse.ArgumentParser(description='Process some integers.')
#parser.add_argument('--output_dir', type=str, help='Output directory for the job')
#args = parser.parse_args()

#output_dir = args.output_dir
#os.makedirs(output_dir, exist_ok=True)
#if not os.path.exists(output_dir):
#    os.makedirs(output_dir)


'Script to run cobaya for fiso for all redshifts'

index = int(sys.argv[1])

# Read the corresponding seed from seeds.txt file
with open("seeds.txt", "r") as f:
    seeds = [int(line.strip()) for line in f]

seed = seeds[index]

rc = False # Not using the current directory as base_path but using instead the no_RC base_path
base_path = '/data/og313/iso/analysis_no_RC'
models42, min_in42, max_in42, taueff_boera_z42 = get_mixed_models(4.2, base_path)#in_file42, out_file42, rc)
models46, min_in46, max_in46, taueff_boera_z46= get_mixed_models(4.6, base_path)#, in_file46, out_file46, rc)
models5, min_in5, max_in5, taueff_boera_z5 = get_mixed_models(5.0, base_path)


log_k = np.arange(-2.2, -0.7, 0.1)

log_k, P_kRC42, P_kRC46, P_kRC5, boera_covariance_matrix = get_data() # %%
P_kRC42 = P_kRC42[:-3]
print('P_kRC42 is', P_kRC42)
P_kRC46 = P_kRC46[:-3]
P_kRC5 = P_kRC5[:-3]

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

tau_T0 = 0.2
tau_gamma = 0.2
tau_u0 = 5
tau_alpha = 0.1
tau_tau_effz42 = taueff_boera_z42
tau_tau_effz46 = taueff_boera_z46
tau_tau_effz5 = taueff_boera_z5

#TEST:
tau_fiso = 0.0025
T0_grid = np.arange(0.5, 1.5, 0.1)
gamma_grid = np.arange(0.9, 1.9, 0.1)
taueff_gridz42 = np.arange(0.3, 1.8, 0.1)*taueff_boera_z42
taueff_gridz46 = np.arange(0.3, 1.8, 0.1)*taueff_boera_z46
taueff_gridz5 = np.arange(0.3, 1.8, 0.1)*taueff_boera_z5

u0_gridz42 = [4.0239696, 4.6623777, 5.6461662, 6.5922353, 7.1101564, 8.1477417, 9.9236799, 11.614156, 13.262641, 15.009738, 18.176741, 21.106591]
u0_gridz46 = [3.6469614, 4.3163684, 5.364912, 6.3899003, 6.589134, 7.687947, 9.587889, 11.434539, 12.536159, 14.414026, 17.83328, 21.078213]
u0_gridz5 = [2.4634363, 3.4376965, 4.3868741, 4.7135148, 5.3831631, 6.3834273, 8.1135775, 9.2448624, 9.917967, 12.360053, 15.563067, 18.783147]
   

diagonal_elements = [tau_T0**2, tau_T0**2, tau_T0**2, tau_gamma**2, tau_gamma**2, tau_gamma**2, (0.104344)**2, ( 0.140348)**2,  (0.184952)**2, tau_u0**2, tau_u0**2,
                    tau_u0**2,tau_fiso**2]

jump_cov = np.diag(diagonal_elements)

#tau cmb interpolator include:
import sys
sys.path.append('/data/emergence5/og313/tau_cmb/IGMtemp_calc/python_routines')
from get_t0_u0_tau import get_params
z_mid_points = [6.5,7,7.5, 8,9,11,13]
alphas = [0.5, 0.6, 0.7, 0.8, 0.9]# 1.0]
x_values = []
tau = []
for z_mid in z_mid_points:
    for alpha in alphas:
        T0arr, u0arr, tau_cmb = get_params(f'zmid_models/zmid_{z_mid:.1f}_alpha_{alpha}_ph.txt')
        #print(u0arr[1], T0arr[1], u0arr[2], T0arr[2],  tau_cmb)
        print(T0arr[2]/(1e4), u0arr[2])
        print(u0arr)
        x_values.append((T0arr[2]/(1e4), u0arr[2]))
        tau.append([tau_cmb])
x_values = np.array(x_values)
tau = np.array(tau).flatten()

# 6. Define new function to predict n any combination of u0_z, T0_z:
def get_tau_cmb(X, a1, a2, a3):

    tau = a1 * X[:, 0] + a2 * X[:, 1] + a3*X[:, 0]*X[:, 1] 
    
    print('----------------------------------')
    print('Extrapolated tau=', tau)
    print('----------------------------------')
    return tau

from scipy.optimize import curve_fit
initial_guesses = [1,1, 1] # one for each coefficient
# lower_bounds = [0] * 6 + [-np.inf]  # a1, a2, ..., c2 ≥ 0, intercept free
# upper_bounds = [np.inf] * 7

params, covariance = curve_fit(
    get_tau_cmb,
    x_values,
    tau,
    p0=initial_guesses
    # bounds=(lower_bounds, upper_bounds)
)

a1_fit, a2_fit, a3_fit= params

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
        'fiso'
    ]
    
    missing_params = [p for p in required_params if p not in params]
    if missing_params:
        print(f"Missing parameters: {missing_params}")
        return -np.inf

    # Group parameters by redshift
    params_by_redshift = [
        [params['T0z42'], params['gammaz42'], params['taueffz42'], 
            params['u0z42'], params['fiso']],
        [params['T0z46'], params['gammaz46'], params['taueffz46'], 
            params['u0z46'], params['fiso']],
        [params['T0z5'], params['gammaz5'], params['taueffz5'], 
            params['u0z5'], params['fiso']]
    ]

    residuals = []
    log_prior_values = []
    redshifts = [4.2, 4.6, 5.0]

    # tau_cmb prior:
    tau_compton = get_tau_cmb(np.column_stack([params['T0z5'], params['u0z5']]), a1_fit, a2_fit, a3_fit)
    tau_planck = 0.054 #Planck+20 vs #Sailer+25
    residual_compton = tau_compton - tau_planck
    sigma_tau_compton = 0.007 #Planck+20
    log_likelihood_tau = -0.5 * np.log(2 * np.pi * sigma_tau_compton**2) - 0.5 * (residual_compton / sigma_tau_compton) ** 2

    
    for i, z in enumerate(redshifts):
        current_params = params_by_redshift[i]

        prediction = get_nn_prediction_mcmc(
            z,
            current_params,
            [min_in42, min_in46, min_in5][i],
            [max_in42, max_in46, max_in5][i],
            [models42, models46, models5][i],
            rc=False
        )
        # Get residuals with kmax cut
        # print('prediction is', prediction)
        prediction = np.array(prediction).flatten()
        prediction = prediction[:-3]
        # print("prediction shape:", prediction.shape)
        # print("target shape:", [P_kRC42, P_kRC46, P_kRC5][i].shape)
        # print('Prediction', np.shape(prediction))
        residuals.append(prediction - [P_kRC42, P_kRC46, P_kRC5][i])

    # Calculate likelihood
    print('Shape of residuals', np.shape(np.array(residuals)))
    residuals = np.array(residuals).flatten()
    cov_matrix_inv = np.linalg.inv(covariance_matrix)

    log_likelihood = -0.5 * np.dot(residuals.T, np.dot(cov_matrix_inv, residuals))
    log_likelihood -= 0.5 * np.log(np.linalg.det(covariance_matrix))
    log_likelihood -= (np.shape(covariance_matrix)[0]) * 0.5 * np.log(2 * np.pi)

    #log_posterior = log_likelihood + sum(log_prior_values)

    log_likelihood_total = log_likelihood_tau + log_likelihood
    return float(log_likelihood_total)



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

# Best-fit samples from T0G run used in tau_cmb paper:
# Best-fit samples [0.90490332, 0.75921842, 0.92943723, 0.94378672, 1.5762943, 1.0759149, 1.0571924, 1.5514588, 1.5563632, 6.2160652, 6.4446825, 3.1331387000000004, 7.005250200000001]
init_vals = [0.90490332, 0.75921842, 0.92943723, 0.94378672, 1.5762943, 1.0759149, 1.0571924, 1.5514588, 1.5563632, 6.2160652, 6.4446825, 3.1331387000000004]
info = {
    'likelihood': {
        'experiment': {
            'external': get_log_likelihood,
            'input_params': [
                'T0z42', 'gammaz42', 'taueffz42', 'u0z42',
                'T0z46', 'gammaz46', 'taueffz46', 'u0z46',
                'T0z5', 'gammaz5', 'taueffz5', 'u0z5', 
                'fiso'
            ],
            'stop_at_error': True
        }
    },
'params': {
    'T0z42': {
        'prior': {'min': 0.5, 'max': 1.4}, 
        'ref': init_vals[0],
        'latex': r'{T_{0}}^{4.2} / {10^{4}} \, \mathrm{(K)}'
    },
    'T0z46': {
        'prior': {'min': 0.5, 'max': 1.4}, 
        'ref': init_vals[1],
        'latex': r'{T_{0}}^{4.6} / {10^{4}} \, \mathrm{(K)}'
    },
    'T0z5': {
        'prior': {'min': 0.5, 'max': 1.4}, 
        'ref': init_vals[2],
        'latex': r'{T_{0}}^{5.0} / {10^{4}} \, \mathrm{(K)}'
    },
    'gammaz42': {
        'prior': {'min': 0.9, 'max': 1.8}, 
        'ref': init_vals[3] ,
        'latex': r'{\gamma}^{\,4.2}'
    },
    'gammaz46': {
        'prior': {'min': 0.9, 'max': 1.8}, 
        'ref': init_vals[4],
        'latex': r'{\gamma}^{\,4.6}'
    },
    'gammaz5': {
        'prior': {'min': 0.9, 'max': 1.8}, 
        'ref': init_vals[5],
        'latex': r'{\gamma}^{\,5.0}'
    },
    'taueffz42': {
        'prior': {'min': 0.31303136056832276, 'max': 1.7738443765538294}, 
        'ref': init_vals[6],
        'latex': r'{\tau_{\mathrm{eff}}}^{\,4.2}'
    },
    'taueffz46': {
        'prior': {'min': 0.42104312690706497, 'max': 2.385911052473369}, 
        'ref': init_vals[7],
        'latex': r'{\tau_{\mathrm{eff}}}^{\,4.6}'
    },
    'taueffz5': {
        'prior': {'min': 0.554854964068882, 'max': 3.1441781297236653}, 
        'ref': init_vals[8],
        'latex': r'{\tau_{\mathrm{eff}}}^{\,5.0}'
    },
    'u0z42': {
        'prior': {'min': 4.0239696, 'max': 21.106591}, 
        'ref': init_vals[9],
        'latex': r'u_{0}^{\,4.2}\,\mathrm{(eV\,m_{p}^{-1})}'
    },
    'u0z46': {
        'prior': {'min': 3.6469614, 'max': 21.078213}, 
        'ref': init_vals[10],
        'latex': r'u_{0}^{\,4.6}\,\mathrm{(eV\,m_{p}^{-1})}'
    },
    'u0z5': {
        'prior': {'min': 2.4634363, 'max': 18.783147}, 
        'ref': init_vals[11],
        'latex': r'u_{0}^{\,5.0}\,\mathrm{(eV\,m_{p}^{-1})}'
    },
    'fiso': {
        'prior': {'min': 0, 'max': 0.01}, 
        'ref': 0,
        'latex': r'f_{\mathrm{iso}}'
    }
},

    'output': f'chains/chain_tau_cmb_kmax_{index}', 
    'sampler': {
        'mcmc': {
            "covmat": jump_cov,
                "covmat_params": [
                'T0z42', 'T0z46', 'T0z5', 
                'gammaz42', 'gammaz46', 'gammaz5',
                'taueffz42', 'taueffz46', 'taueffz5',
                'u0z42', 'u0z46', 'u0z5',
                'fiso'
            ],
            "seed": 45,
            "learn_proposal": True,
            "Rminus1_stop": 0.005,
            "Rminus1_cl_stop": 0.04,
            "oversample_power": 0.5,
            # 'max_samples':np.inf,
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
