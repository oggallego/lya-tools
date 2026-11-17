# data_processing.py

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import NNInterpolator
from config import redshift_configs
import os
import ast 
import pandas as pd

def get_rc(cosmology, z):
    """
    Generate a nested dictionary for Rs10 correction for u0 and taueff for a given cosmology and redshift.
    """

    taueff_input = np.arange(0.3, 1.8, 0.1)
    u0vals = np.arange(0,12,1)

    # Create a nested dictionary
    rc_cdm = {z: {u0_idx: {round(i, 2): [] for i in taueff_input} for u0_idx in u0vals}}
    # print(rc_cdm)

    #Need information from u0 table:
    column_names = ['Model', 'zr', 'tau_e', 'u0_4.2<=z<=12.0', 'u0_4.6<=z<=13.0', 'u0_6.0<=z<=13.0']
    u0all = pd.read_csv('/data/og313b/lyalpha/taue_u0_table.txt', skiprows=4, delim_whitespace=True, names=column_names)
    if z == 4.2:
        all_u0z = u0all.sort_values(column_names[3]).iloc[:, [0, 3]].reset_index(drop=True)
        print(all_u0z)
        model_names = all_u0z.iloc[:, 0]
    elif z == 4.6:
        all_u0z = u0all.sort_values(column_names[4]).iloc[:, [0, 4]].reset_index(drop=True)
        model_names = all_u0z.iloc[:, 0]
    elif z == 5.0:
        all_u0z = u0all.sort_values(column_names[5]).iloc[:, [0, 5]].reset_index(drop=True)

    else:
        raise ValueError(f"Unsupported value of z: {z}")

    model_names = all_u0z.iloc[:, 0]

    fid = 'planck1_20_1024'


    # model_names = all_u0z.iloc[:, 0]
    # Populate rc_cdm with models

    for model_name in model_names:

        # Get the corresponding u0 value for the current model_name
        u0 = all_u0z[all_u0z.iloc[:, 0] == model_name].iloc[0, 1]
        # Get the index of the model
        model_index = model_names[all_u0z.iloc[:, 1] == u0].index[0]
    #     print(f"Index of model with u0={u0}: {model_index}")
    # except IndexError:
    #     print(f"Model with u0={u0} not found in the list.")
    #     continue  # S
        for i in taueff_input:
            taueff_key = round(i, 2)
            suffix = model_name.split('1024')[1]
            if 'wdm' in cosmology:
                model = np.loadtxt(f"/data/og313b/lyalpha/models/{cosmology}/{fid}_wdm{cosmology.split('wdm')[1]}{suffix}/Rs10_{z}{suffix}_{round(i, 2)}.txt")
            if cosmology == 'cdm':
                model = np.loadtxt(f"/data/og313b/lyalpha/models/{cosmology}/{fid}{suffix}/Rs10_{z}{suffix}_{round(i, 2)}.txt")
            rc_cdm[z][model_index][taueff_key].append(model)

    return rc_cdm  

'For mixed models - in interpolation in fwdm plane as well'
def get_rc_mixed(cosmology, z):


    taueff_input = np.arange(0.3, 1.8, 0.1)
    u0vals = np.arange(0,12,1)

    # Create a nested dictionary
    rc = {z: {u0_idx: {round(i, 2): [] for i in taueff_input} for u0_idx in u0vals}}
    # print(rc_cdm)

    #Need information from u0 table:
    column_names = ['Model', 'zr', 'tau_e', 'u0_4.2<=z<=12.0', 'u0_4.6<=z<=13.0', 'u0_6.0<=z<=13.0']
    u0all = pd.read_csv('/data/og313b/lyalpha/taue_u0_table.txt', skiprows=4, delim_whitespace=True, names=column_names)
    if z == 4.2:
        all_u0z = u0all.sort_values(column_names[3]).iloc[:, [0, 3]].reset_index(drop=True)
        print(all_u0z)
        model_names = all_u0z.iloc[:, 0]
    elif z == 4.6:
        all_u0z = u0all.sort_values(column_names[4]).iloc[:, [0, 4]].reset_index(drop=True)
        model_names = all_u0z.iloc[:, 0]
    elif z == 5.0:
        all_u0z = u0all.sort_values(column_names[5]).iloc[:, [0, 5]].reset_index(drop=True)

    else:
        raise ValueError(f"Unsupported value of z: {z}")

    model_names = all_u0z.iloc[:, 0]

    fid = 'planck1_20_1024'


    # model_names = all_u0z.iloc[:, 0]
    # Populate rc_cdm with models

    for model_name in model_names:

        # Get the corresponding u0 value for the current model_name
        u0 = all_u0z[all_u0z.iloc[:, 0] == model_name].iloc[0, 1]
        # Get the index of the model
        model_index = model_names[all_u0z.iloc[:, 1] == u0].index[0]
    #     print(f"Index of model with u0={u0}: {model_index}")
    # except IndexError:
    #     print(f"Model with u0={u0} not found in the list.")
    #     continue  # S
        for i in taueff_input:
            taueff_key = round(i, 2)
            suffix = model_name.split('1024')[1]
            
            model = np.loadtxt(f"/data/og313b/lyalpha/models/{cosmology}/{fid}_fwdm{cosmology.split('_')[0][1:]}_m{cosmology.split('_')[1]}.0{suffix}/Rs10_{z}{suffix}_{round(i, 2)}.txt")
            rc[z][model_index][taueff_key].append(model)

    return rc



def extract_params_from_filename(filename): 
    """
    Extract parameters T0, gamma, and taueff from a filename.
    """
    parts = filename.split('_')

    T0 = float(parts[4])
    gamma = float(parts[6])
    taueff = float(parts[7])
    return T0, gamma, taueff

def read_ps_file(file_path, z): # now valid for all z
    """
    Extract u0 and power spectrum values from ps files.
    """
    with open(file_path, 'r') as file:
        u0 = None
        ps_values = []
        for line in file:
            print
            if z == 4.2:
                # The strings below are part of the header information in each ps file:
                if 'u0 [eV/mp] (4.2<=z<=12.0)' in line:
                    # u0 grid in ascending order for that redshift:
                    u0_values = ast.literal_eval(line.split(':')[1].strip())
                    # print(u0_values)
                elif 'u0 idx in u0(z) grid' in line:
                    # print(ast.literal_eval(line.split(':')[1].strip()))
                    u0_idx = ast.literal_eval(line.split(':')[1].strip())[0]  # First index for z=4.2
                    # Get the index from this linein the header information
                    u0 = u0_values[u0_idx]
                    # u0 corresponding to that index
                    # print(u0)
                # Lines that do not contain : are not part of the header and contain the ps information:
                elif not ':' in line:
                    ps_values.append(float(line.split()[2]))
                    ps_valuesz = np.array(ps_values)[:16]
            if z == 4.6:
                if 'u0 [eV/mp] (4.6<=z<=13.0)' in line:
                    u0_values = ast.literal_eval(line.split(':')[1].strip())
                    # print(u0_values)
                elif 'u0 idx in u0(z) grid' in line:
                    u0_idx = ast.literal_eval(line.split(':')[1].strip())[1] 
                    u0 = u0_values[u0_idx]
                    # print(u0)
                elif not ':' in line:
                    ps_values.append(float(line.split()[2]))
                    ps_valuesz = np.array(ps_values)[16:32]

            if z == 5.0:
                if 'u0 [eV/mp] (6.0<=z<=13.0)' in line:
                    u0_values = ast.literal_eval(line.split(':')[1].strip())
                elif 'u0 idx in u0(z) grid' in line:
                    u0_idx = ast.literal_eval(line.split(':')[1].strip())[2]  # Assuming we're using the first index for z=4.2
                    u0 = u0_values[u0_idx]
                elif not ':' in line:
                    ps_values.append(float(line.split()[2]))
                    ps_valuesz = np.array(ps_values)[32:]

    return u0, u0_idx, ps_valuesz  # return u0 and ps_values for that redshift

def read_data_nn(base_path, in_file, out_file):

    """
    Reads and processes input and output data for neural network training.

    Args:
        base_path (str): Base path to the data directories.
        in_file (str): Name of the input file.
        out_file (str): Name of the output file.

    Returns:
        tuple: Processed input array and output array.
    """

    
    # Load the original arrays
    in_cdm = np.loadtxt(f'{base_path}/cdm/{in_file}')
    in_wdm4_f0125 = np.loadtxt(f'{base_path}/f0.125_wdm4/{in_file}')
    in_wdm4_f025 = np.loadtxt(f'{base_path}/f0.25_wdm4/{in_file}')
    in_wdm4_f05 = np.loadtxt(f'{base_path}/f0.5_wdm4/{in_file}')
    in_wdm4 = np.loadtxt(f'{base_path}/wdm4/{in_file}')

    in_wdm3_f0125 = np.loadtxt(f'{base_path}/f0.125_wdm3/{in_file}')
    in_wdm3_f025 = np.loadtxt(f'{base_path}/f0.25_wdm3/{in_file}')
    in_wdm3_f05 = np.loadtxt(f'{base_path}/f0.5_wdm3/{in_file}')
    in_wdm3 = np.loadtxt(f'{base_path}/wdm3/{in_file}')

    in_wdm2_f0125 = np.loadtxt(f'{base_path}/f0.125_wdm2/{in_file}')
    in_wdm2_f025 = np.loadtxt(f'{base_path}/f0.25_wdm2/{in_file}')
    in_wdm2_f05 = np.loadtxt(f'{base_path}/f0.5_wdm2/{in_file}')
    in_wdm2 = np.loadtxt(f'{base_path}/wdm2/{in_file}')

    in_wdm1_f0125 = np.loadtxt(f'{base_path}/f0.125_wdm1/{in_file}')
    in_wdm1_f025 = np.loadtxt(f'{base_path}/f0.25_wdm1/{in_file}')
    in_wdm1_f05 = np.loadtxt(f'{base_path}/f0.5_wdm1/{in_file}')
    in_wdm1 = np.loadtxt(f'{base_path}/wdm1/{in_file}')

    # Define the fixed parameter alpha for each array
    alpha_cdm = 0
    alpha_wdm4 = 1/4
    alpha_wdm3 = 1/3
    alpha_wdm2 = 1/2
    alpha_wdm1 = 1

    column_alpha_cdm = np.full((in_cdm.shape[0], 1), alpha_cdm)
    column_alpha_wdm4 = np.full((in_wdm4.shape[0], 1), alpha_wdm4)
    column_alpha_wdm3 = np.full((in_wdm3.shape[0], 1), alpha_wdm3)
    column_alpha_wdm2 = np.full((in_wdm2.shape[0], 1), alpha_wdm2)
    column_alpha_wdm1 = np.full((in_wdm1.shape[0], 1), alpha_wdm1)

    column_f1 = np.full((in_cdm.shape[0], 1), 1)
    column_f0125 = np.full((in_cdm.shape[0], 1), 0.125)
    column_f025 = np.full((in_cdm.shape[0], 1), 0.25)
    column_f05 = np.full((in_cdm.shape[0], 1), 0.5)
    column_f0 = np.full((in_cdm.shape[0], 1), 0)


    new_cdm = np.concatenate((in_cdm, column_alpha_cdm, column_f0), axis=1)

    'Along alpha axis, f = 0'
    new_wdm4_f0 = np.concatenate((in_cdm, column_alpha_wdm4, column_f0), axis=1)
    new_wdm3_f0 = np.concatenate((in_cdm, column_alpha_wdm3, column_f0), axis=1)
    new_wdm2_f0 = np.concatenate((in_cdm, column_alpha_wdm2, column_f0), axis=1)
    new_wdm1_f0 = np.concatenate((in_cdm, column_alpha_wdm1, column_f0), axis=1)


    'Along f axis, alpha = 0'
    new_cdm_f0125 = np.concatenate((in_cdm, column_alpha_cdm, column_f0125), axis=1)
    new_cdm_f025 = np.concatenate((in_cdm, column_alpha_cdm, column_f025), axis=1)
    new_cdm_f05 = np.concatenate((in_cdm, column_alpha_cdm, column_f05), axis=1)
    new_cdm_f1 = np.concatenate((in_cdm, column_alpha_cdm, column_f1), axis=1)
    # print(np.where((new_cdm_f05[4]!=0)))
    # print(new_cdm_f05[1])

    new_wdm4 = np.concatenate((in_wdm4, column_alpha_wdm4, column_f1), axis=1)
    new_wdm4_f0125 = np.concatenate((in_wdm4_f0125, column_alpha_wdm4, column_f0125), axis=1)
    new_wdm4_f025 = np.concatenate((in_wdm4_f025, column_alpha_wdm4, column_f025), axis=1)
    new_wdm4_f05 = np.concatenate((in_wdm4_f05, column_alpha_wdm4, column_f05), axis=1)

    new_wdm3 = np.concatenate((in_wdm3, column_alpha_wdm3, column_f1), axis=1)
    new_wdm3_f0125 = np.concatenate((in_wdm3_f0125, column_alpha_wdm3, column_f0125), axis=1)
    new_wdm3_f025 = np.concatenate((in_wdm3_f025, column_alpha_wdm3, column_f025), axis=1)
    new_wdm3_f05 = np.concatenate((in_wdm3_f05, column_alpha_wdm3, column_f05), axis=1)

    new_wdm2 = np.concatenate((in_wdm2, column_alpha_wdm2, column_f1), axis=1)
    new_wdm2_f0125 = np.concatenate((in_wdm2_f0125, column_alpha_wdm2, column_f0125), axis=1)
    new_wdm2_f025 = np.concatenate((in_wdm2_f025, column_alpha_wdm2, column_f025), axis=1)
    new_wdm2_f05 = np.concatenate((in_wdm2_f05, column_alpha_wdm2, column_f05), axis=1)

    #ADD WDM1 here:
    new_wdm1 = np.concatenate((in_wdm1, column_alpha_wdm1, column_f1), axis=1)
    new_wdm1_f0125 = np.concatenate((in_wdm1_f0125, column_alpha_wdm1, column_f0125), axis=1)
    new_wdm1_f025 = np.concatenate((in_wdm1_f025, column_alpha_wdm1, column_f025), axis=1)
    new_wdm1_f05 = np.concatenate((in_wdm1_f05, column_alpha_wdm1, column_f05), axis=1)

    # Final shape is 234000, 6 for 6 input parameters 
    final_input_array = np.concatenate((new_cdm,
                                        new_wdm4_f0, new_wdm3_f0, new_wdm2_f0, new_wdm1_f0,
                                        new_cdm_f0125, new_cdm_f025, new_cdm_f05, new_cdm_f1,
                                        new_wdm4_f0125, new_wdm4_f025, new_wdm4_f05, new_wdm4,
                                        new_wdm3_f0125, new_wdm3_f025, new_wdm3_f05, new_wdm3,
                                        new_wdm2_f0125, new_wdm2_f025, new_wdm2_f05, new_wdm2,
                                        new_wdm1_f0125, new_wdm1_f025, new_wdm1_f05, new_wdm1), axis=0).reshape(-1, 6)
    
    'ps files for mixed models names _RC include both the RC and the CAMB/CLASS correction'


    ps_cdm = np.loadtxt(f'{base_path}/cdm/{out_file}')

    ps_wdm4_f0125 = np.loadtxt(f'{base_path}/f0.125_wdm4/{out_file}')
    ps_wdm4_f025 = np.loadtxt(f'{base_path}/f0.25_wdm4/{out_file}')
    ps_wdm4_f05 = np.loadtxt(f'{base_path}/f0.5_wdm4/{out_file}')
    ps_wdm4 = np.loadtxt(f'{base_path}/wdm4/{out_file}')

    ps_wdm3_f0125 = np.loadtxt(f'{base_path}/f0.125_wdm3/{out_file}')
    ps_wdm3_f025 = np.loadtxt(f'{base_path}/f0.25_wdm3/{out_file}')
    ps_wdm3_f05 = np.loadtxt(f'{base_path}/f0.5_wdm3/{out_file}')
    ps_wdm3 = np.loadtxt(f'{base_path}/wdm3/{out_file}')

    ps_wdm2_f0125 = np.loadtxt(f'{base_path}/f0.125_wdm2/{out_file}')
    ps_wdm2_f025 = np.loadtxt(f'{base_path}/f0.25_wdm2/{out_file}')
    ps_wdm2_f05 = np.loadtxt(f'{base_path}/f0.5_wdm2/{out_file}')
    ps_wdm2 = np.loadtxt(f'{base_path}/wdm2/{out_file}')

    ps_wdm1_f0125 = np.loadtxt(f'{base_path}/f0.125_wdm1/{out_file}')
    ps_wdm1_f025 = np.loadtxt(f'{base_path}/f0.25_wdm1/{out_file}')
    ps_wdm1_f05 = np.loadtxt(f'{base_path}/f0.5_wdm1/{out_file}')
    ps_wdm1 = np.loadtxt(f'{base_path}/wdm1/{out_file}')

    # Final shape of ps array: 1500*12*13, 16 for 16 output parameters
    final_ps_array= np.concatenate((ps_cdm,
                                    ps_cdm, ps_cdm, ps_cdm, ps_cdm,
                                    ps_cdm, ps_cdm, ps_cdm, ps_cdm,
                                    ps_wdm4_f0125 , ps_wdm4_f025 , ps_wdm4_f05 ,ps_wdm4, 
                                    ps_wdm3_f0125 , ps_wdm3_f025 , ps_wdm3_f05, ps_wdm3,
                                    ps_wdm2_f0125 , ps_wdm2_f025 , ps_wdm2_f05, ps_wdm2,
                                    ps_wdm1_f0125 , ps_wdm1_f025 , ps_wdm1_f05, ps_wdm1), axis = 0)


    # Verify the shape of the final array
    print('Shape of final input array', np.shape(final_input_array))  
    print('Shape of final labels array', np.shape(final_ps_array)) 
    
    return final_input_array, final_ps_array



def get_train_test_data(z, base_path, batch_size, patience, patience_scheduler):
    """
    Get training and test data for the given redshift and save preprocessed data.
    """
    # inputs, labels = read_data_nn(base_path, in_file, out_file)

    print('Using base path', base_path) # This is defined in run cobaya script to use the current working directory
    inputs = np.loadtxt(f'{base_path}/z{z}/inputs{z}{base_path.split("analysis")[1]}_all.txt')
    print(f'Inputs are inputsz{base_path.split("analysis")[1]}_all.txt')

    labels = np.loadtxt(f'{base_path}/z{z}/ps{z}{base_path.split("analysis")[1]}_all.txt')
    print('Shape of all inputs loaded', np.shape(labels))
    print('Shape of all inputs loaded', np.shape(inputs))
    config = redshift_configs[z]
    
    model = NNInterpolator(
        input_size=inputs.shape[1],
        output_size=labels.shape[1],
        hidden=120,
        learning_rate=0.0001,  # Fixed learning rate
        act=torch.nn.ReLU,
        patience=config["patience"],
        patience_scheduler=config["patience_scheduler"],
        base_path = base_path

    )

    # Preprocess the data and saves train and test files:
    train_loader, test_loader, min_in, max_in = model.preprocess_data(
        z, inputs, np.log(labels), batch_size=batch_size)
    
def load_preprocessed_data(model, z):
    """
    Load preprocessed data from a .npz file, using the model's output path logic.

    Args:
        model (NNInterpolator): An instance of the NNInterpolator class.
        z (float): Redshift value.

    Returns:
        tuple: Preprocessed input (x) and output (y) arrays.
    """
    # Get the output path using the model's method
    output_path = model.get_output_path(z)

    # Path to the preprocessed data file
    train_data_path = os.path.join(output_path, "train_data.npz")

    # Load the preprocessed data
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"Preprocessed data not found at {train_data_path}")

    data = np.load(train_data_path)
    return data["x"], data["y"]

def save_data(target_redshift):

    z = target_redshift

    config = redshift_configs[target_redshift]


    base_path=config["base_path"]
    in_file=config["in_file"]
    out_file=config["out_file"]
    print(f'Saving data from {base_path}', in_file, out_file)

    final_input_array, final_ps_array = read_data_nn(base_path, in_file, out_file)

    np.savetxt(f'{base_path}/z{z}/{in_file.split(".txt")[0]}_all.txt', final_input_array)
    print(f'Have saved in {base_path}')
    np.savetxt(f'{base_path}/z{z}/{out_file.split(".txt")[0]}_all.txt', final_ps_array)

    print(f'Saved inputs and outputs for all models for z = {z}')
