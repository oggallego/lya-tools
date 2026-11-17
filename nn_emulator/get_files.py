
#%%
import numpy as np
import ast
import os
import argparse
from tqdm import tqdm
import sys
sys.path.append('/data/og313/lyalpha/code_wdm/nn_emulator_fwdm')
from data_processing import get_rc, extract_params_from_filename, read_ps_file

from config import redshift_configs

def get_files(cosmology, z):

    """
    For a given base_path in config.py, (same directory as where thist file is being) create cosmology directories with inputs and ps data
    In particular, this file is for generating input and output including resolution correction for CDM.
    Mixed models include camb/class correction.

    Output files in same directory as base_path as defined by config.py
    """

    base_files_path = '/data/og313b/lyalpha/models'
    inputs = []
    outputs = []

    # Get rc dictio
    # nary for this cosmology and z:
    # NOTE: we use the resolution correction for CDM - would be interesting to see how this compares if interpolating in fwdm plane

    
    rc = get_rc('cdm', z)

    T0_grid = np.arange(0.5, 1.5, 0.1)*10**4
    gamma_grid = np.arange(0.9, 1.9, 0.1)
    taueff_grid = np.arange(0.3, 1.8, 0.1)


    # # READS THE PS FILES PER REDSHIFT:
    # base_path = os.path.join(base_path, cosmology)


    # Directory where it will be saved:
    config = redshift_configs[z]
    directory = os.path.join(config["base_path"], cosmology)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get tau_eff at that redshift:
    taueff_boera_z = 1.56 * ((1 + z) / 5.75) ** 4
    
    base_path = os.path.join(base_files_path, cosmology)

    for root, dirs, files in os.walk(base_path):
        files.sort()
        # Only access /ps_files? directories (avoid other thest directories):
        if 'ps_files' in root and 'vcold' not in root and 'test' not in root and '10_1024' not in root and '10_512' not in root and '4.2_4.6' not in root and 'new' not in root and '_old' not in root and 'cold_z42' not in root and 'rerun' not in root:
            print('Order in which simulationsa are read', root)
            for file in tqdm(files, desc="Processing files", unit="file"):
                if file.startswith('ps'):
                    file_path = os.path.join(root, file)
                    T0, gamma, taueff = extract_params_from_filename(file)
                    taueff_scaled = taueff * taueff_boera_z #input data used for NN is original taueff
                    # Get u0 and ps for that redshift
                    u0, u0_idx, ps_values = read_ps_file(file_path, z)
                    if u0 is not None:
                    
                        T0_idx = int(np.where(np.isclose(T0_grid, T0))[0][0])
                        gamma_idx = int(np.where(np.isclose(gamma_grid, gamma))[0][0])
                        taueff_idx = int(np.where(np.isclose(taueff_grid, taueff))[0][0])

                        camb_class_corr = np.loadtxt(f'/data/og313b/lyalpha/models/cdm_class/camb_class_cdm_ratio/corr_u0{u0_idx}_{T0_idx}_{gamma_idx}_{taueff_idx}_z{z}.txt')
                        # lenght of each of these files is 16
                        
                        inputs.append([T0, gamma, taueff_scaled, u0])

                        if 'f' in cosmology:
                            # print('Mixed model: including CAMB/CLASS correction')
                            outputs.append(ps_values*rc[z][u0_idx][taueff]*camb_class_corr)
                        if 'f' not in cosmology:
                            # print('Pure model: only resolution correctio')
                            outputs.append(ps_values*rc[z][u0_idx][taueff])
                    

    inputs = np.array(inputs)
    print(np.shape(outputs))
    outputs = np.array(outputs).reshape(18000,16)
    print('Inputs, outputs ready with shape', np.shape(inputs), np.shape(outputs))

    # Save inputs and output for each model:
    np.savetxt(f'{directory}/ps{config["out_file"].split("ps")[1]}', outputs)
    np.savetxt(f'{directory}/inputs{config["out_file"].split("ps")[1]}', inputs)

#     print(f'Saved outputs and inputs for all models for {cosmology} at z = {z}')

cosmologies = ['cdm', 'wdm4', 'wdm3', 'wdm2',
               'f0.5_wdm4', 'f0.25_wdm4', 'f0.125_wdm4', 
               'f0.5_wdm3', 'f0.25_wdm3', 'f0.125_wdm3',
                 'f0.5_wdm2','f0.25_wdm2',  'f0.125_wdm2']

parser = argparse.ArgumentParser(description='Process some z values.')
parser.add_argument('--z', type=float, required=True)
args = parser.parse_args()
z = args.z


#TEST:
# z = 4.2    
# get_files_rc_cdm_u0('cdm', z)
# cosmologies = ['cdm']
# z_values= [4.2]
for cosmo in cosmologies:
    get_files(cosmo, z)

