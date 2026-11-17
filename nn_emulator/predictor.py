import numpy as np
import torch
from scipy.interpolate import interp1d

def nn_interpolation(new_input, min_in, max_in, model):
    """
    Perform NN interpolation for a single model.

    Args:
        new_input (np.ndarray): Input data for prediction.
        min_in (np.ndarray): Minimum values of input features (for normalization).
        max_in (np.ndarray): Maximum values of input features (for normalization).
        model (NNInterpolator): Trained model.
    Returns:
        np.ndarray: Prediction from the model.
    """
    # Normalize the input
    new_input = (new_input - min_in) / (max_in - min_in)
    new_input = torch.from_numpy(np.array(new_input)).float()

    with torch.no_grad():
        # prediction = (model.model(new_input)).numpy()

        # Get the prediction
        prediction = model.model(new_input)

        # Ensure the prediction is a tensor and process it correctly
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.cpu().detach().numpy()  # Move to CPU and convert to numpy
        else:
            raise TypeError("Output of model.model(new_input) is not a tensor")



        prediction = np.exp(prediction)  # Reverse log transformation

    return prediction


def get_nn_prediction(z, new_input, min_in, max_in, models, rc):
    """
    Get predictions for a given input using all models for the specified redshift.

    Args:
        z (float): Redshift value.
        new_input (np.ndarray): Input data for prediction.
        min_in (np.ndarray): Minimum values of input features.
        max_in (np.ndarray): Maximum values of input features.
        models (list): List of trained models for the redshift.

    Returns:
        np.ndarray: Final predictions after applying BC correction.
    """
    # Define a mapping of redshifts to BC correction file paths
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

    # Ensure the redshift has a corresponding BC correction file
    if z not in bc_cdm_paths:
        raise ValueError(f"No BC correction file path defined for redshift {z}")

    # Get the BC correction file path for the redshift
    bc_cdm_path = bc_cdm_paths[z]
    bc_cdm = np.loadtxt(bc_cdm_path)
    
    rc_cdm_path = rc_cdm_paths[z]
    rc_cdm = np.loadtxt(rc_cdm_path)
    # new_input[0] = new_input[0] * 1e4
    # taueff_boera_z = 1.56 * ((1 + z) / 5.75) ** 4
    # new_input[2] = new_input[2] / taueff_boera_z

    # Predict using all models
    num_models = len(models)
  
    new_out = np.zeros((np.shape(new_input)[0], num_models, 16))
    avg_preds = []
    for i in range(np.shape(new_input)[0]):
        print('Input going into the NN', new_input[i])
        for j in range(num_models):
            new_out[i, j] = nn_interpolation(new_input[i], min_in, max_in, models[j])
    avg_preds = np.mean(new_out, axis=1)

    # Average prediction
    if rc == True:
        print('Inlcuding both box correction and resolution correction!')
        avg_preds_final = avg_preds * bc_cdm * rc_cdm
    elif rc == False:
        print('Now including box correction!')
        avg_preds_final = avg_preds * bc_cdm
    return avg_preds_final

def get_nn_prediction_mcmc(z, new_input, min_in, max_in, models, rc):
    """
    Get predictions for a given input using single model obtained using train.py for the specified redshift.
    MCMCM is run with scaled T0 - so need to re-scale T0 back into original (in which NN is trained)
    Args:
        z (float): Redshift value.
        new_input (np.ndarray): Input data for prediction.
        min_in (np.ndarray): Minimum values of input features.
        max_in (np.ndarray): Maximum values of input features.
        models (list): List of trained models for the redshift.

    Returns:
        np.ndarray: Final predictions after applying BC correction.
    """

    new_input[0] = new_input[0] * (10**4)
    
    # Define a mapping of redshifts to BC correction file paths
    bc_cdm_paths = {
        4.2: "/data/og313b/lyalpha/models/cdm/planck1_20_1024/bc_4.2_1.0.txt",
        4.6: "/data/og313b/lyalpha/models/cdm/planck1_20_1024/bc_4.6_1.0.txt",
        5.0: "/data/og313b/lyalpha/models/cdm/planck1_20_1024/bc_5.0_1.0.txt"
    }

    rc_cdm_paths = {
        4.2: "/data/og313b/lyalpha/models/cdm/planck1_20_1024/Rs10_4.2_1.0.txt",
        4.6: "/data/og313b/lyalpha/models/cdm/planck1_20_1024/Rs10_4.6_1.0.txt",
        5.0: "/data/og313b/lyalpha/models/cdm/planck1_20_1024/Rs10_5.0_1.0.txt"
    }

    # Ensure the redshift has a corresponding BC correction file
    if z not in bc_cdm_paths:
        raise ValueError(f"No BC correction file path defined for redshift {z}")

    # Get the BC correction file path for the redshift
    bc_cdm_path = bc_cdm_paths[z]
    bc_cdm = np.loadtxt(bc_cdm_path)

    rc_cdm_path = rc_cdm_paths[z]
    rc_cdm = np.loadtxt(rc_cdm_path)

    # new_input[0] = new_input[0] * 1e4
    # taueff_boera_z = 1.56 * ((1 + z) / 5.75) ** 4
    # new_input[2] = new_input[2] / taueff_boera_z

    # Predict using all models
    num_models = len(models)
    new_out = np.zeros((num_models, 16))
    avg_preds = []
    for j in range(num_models):
        new_out[j] = nn_interpolation(new_input, min_in, max_in, models[j])
    avg_preds.append(np.mean(new_out, axis=0))
    # Average prediction
    if rc == True:
        avg_preds_final = avg_preds * bc_cdm * rc_cdm
    if rc == False:
        avg_preds_final = avg_preds * bc_cdm
    if rc == None:
        avg_preds_final = avg_preds
    
        
    return avg_preds_final



def get_nn_prediction_mcmc_noise(z, new_input, min_in, max_in, models):
    """
    Get predictions for a given input using single model obtained using train.py for the specified redshift.
    MCMCM is run with scaled T0 - so need to re-scale T0 back into original (in which NN is trained)
    Args:
        z (float): Redshift value.
        new_input (np.ndarray): Input data for prediction.
        min_in (np.ndarray): Minimum values of input features.
        max_in (np.ndarray): Maximum values of input features.
        models (list): List of trained models for the redshift.

    Returns:
        np.ndarray: Final predictions after applying BC correction.
    """

    new_input[0] = new_input[0] * (10**4)
    
    num_models = len(models)
    new_out = np.zeros((num_models, 16))
    avg_preds = []
    for j in range(num_models):
        new_out[j] = nn_interpolation(new_input, min_in, max_in, models[j])
    avg_preds.append(np.mean(new_out, axis=0))
  
    avg_preds_final = avg_preds
    
        
    return avg_preds_final

#predictor.py
from scipy.interpolate import interp2d

# Load the 3D arrays - include in cobaya script so that it is done once:
Rs10_42 = np.load('/data/og313/lyalpha/code_wdm/analysis_RC_cdm_u0_post/Rs10_42.npy')
Rs10_46 = np.load('/data/og313/lyalpha/code_wdm/analysis_RC_cdm_u0_post/Rs10_46.npy')
Rs10_5 = np.load('/data/og313/lyalpha/code_wdm/analysis_RC_cdm_u0_post/Rs10_5.npy')

spectrum_centers = np.arange(-2.2, -0.7, 0.1)


def get_nn_prediction_with_interpolation(z, new_input, min_in, max_in, models, u042, u046, u05, Rs10_42, Rs10_46, Rs10_5):
    """
    Get predictions for a given input using single model obtained using train.py for the specified redshift.
    MCMCM is run with scaled T0 - so need to re-scale T0 back into original (in which NN is trained)
    Args:
        z (float): Redshift value.
        new_input (np.ndarray): Input data for prediction.
        min_in (np.ndarray): Minimum values of input features.
        max_in (np.ndarray): Maximum values of input features.
        models (list): List of trained models for the redshift.

    Returns:
        np.ndarray: Final predictions after applying BC correction.
    """

    new_input[0] = new_input[0] * (10**4)
    
    # Define a mapping of redshifts to BC correction file paths
    bc_cdm_paths = {
        4.2: "/data/og313b/lyalpha/models/cdm/planck1_20_1024/bc_4.2_1.0.txt",
        4.6: "/data/og313b/lyalpha/models/cdm/planck1_20_1024/bc_4.6_1.0.txt",
        5.0: "/data/og313b/lyalpha/models/cdm/planck1_20_1024/bc_5.0_1.0.txt"
    }


    # Ensure the redshift has a corresponding BC correction file
    if z not in bc_cdm_paths:
        raise ValueError(f"No BC correction file path defined for redshift {z}")

    # Get the BC correction file path for the redshift
    bc_cdm_path = bc_cdm_paths[z]
    bc_cdm = np.loadtxt(bc_cdm_path)


    Rs10z_interp1 = np.zeros((len(spectrum_centers)))

    #The grid of scaling factors is in terms of tau_efffBoera:
    # taueff = 1.56 * ((1 + z) / 5.75) ** 4 #Boera+19
    # scaling_factors_boera = taueff*scaling_factors

    # Perform bilinear interpolation
    for i in range(len(spectrum_centers)):

        if z == 4.2:
            for i in range(len(spectrum_centers)): #idx = 7 for fixing correction at tau_eff = 1
                Rs10z_interp1 [:, i] = interp1d(u042, Rs10_42[:, 7, i])(new_input[3])

            # # Create the 2D interpolation function
            # interp_func = interp2d(u042, scaling_factors_boera, Rs10_42[:, :, i].T, kind='linear')
            # # Interpolate at the desired points
            # Rs10z_interp1[0, :, i] = interp_func(new_input[3], new_input[2]) #u0,z = 4.2, tau_eff, z = 4.2 indices
        
    
        elif z == 4.6:  
            for i in range(len(spectrum_centers)): #idx = 7 for fixing correction at tau_eff = 1
                Rs10z_interp1 [:, i] = interp1d(u046, Rs10_46[:, 7, i])(new_input[3])

        elif z == 5.0:
            for i in range(len(spectrum_centers)): #idx = 7 for fixing correction at tau_eff = 1
                Rs10z_interp1 [:, i] = interp1d(u05, Rs10_5[:, 7, i])(new_input[3])
    
    #Rs10z_interp1.reshape(-1) # 16 bins array including Rs for that u0 and tau_eff
            

    # Predict using all models
    num_models = len(models)
    new_out = np.zeros((num_models, 16))
    avg_preds = []
    for j in range(num_models):
        new_out[j] = nn_interpolation(new_input, min_in, max_in, models[j])
    avg_preds.append(np.mean(new_out, axis=0))
    # Average prediction
 
    avg_preds_final = avg_preds * bc_cdm * Rs10z_interp1
  

        
    return avg_preds_final

# How would the prediction need to change?

def get_nn_prediction_ratio(z, new_input, min_in, max_in, models, rc):
    num_models = len(models)
    print('How many models?', num_models)
    if num_models == 0:
        raise ValueError("No models found in models_ratio")
    
    # Load Pk, fid at each redshift:
    pkz_fid = np.loadtxt(f'/data/og313b/lyalpha/models/cdm/planck1_20_1024/ps_fidz{z}.txt')

    # # Rescale inputs:
    # new_input[0] = new_input[0] * (10**4)
    
    # Define a mapping of redshifts to BC correction file paths
    bc_cdm_paths = {
        4.2: "/data/og313b/lyalpha/models/cdm/planck1_20_1024/bc_4.2_1.0.txt",
        4.6: "/data/og313b/lyalpha/models/cdm/planck1_20_1024/bc_4.6_1.0.txt",
        5.0: "/data/og313b/lyalpha/models/cdm/planck1_20_1024/bc_5.0_1.0.txt"
    }

    rc_cdm_paths = {
        4.2: "/data/og313b/lyalpha/models/cdm/planck1_20_1024/Rs10_4.2_1.0.txt",
        4.6: "/data/og313b/lyalpha/models/cdm/planck1_20_1024/Rs10_4.6_1.0.txt",
        5.0: "/data/og313b/lyalpha/models/cdm/planck1_20_1024/Rs10_5.0_1.0.txt"
    }

    # Ensure the redshift has a corresponding BC correction file
    if z not in bc_cdm_paths:
        raise ValueError(f"No BC correction file path defined for redshift {z}")

    # Get the BC correction file path for the redshift
    bc_cdm_path = bc_cdm_paths[z]
    bc_cdm = np.loadtxt(bc_cdm_path)

    rc_cdm_path = rc_cdm_paths[z]
    rc_cdm = np.loadtxt(rc_cdm_path)

    # new_input[0] = new_input[0] * 1e4
    # taueff_boera_z = 1.56 * ((1 + z) / 5.75) ** 4
    # new_input[2] = new_input[2] / taueff_boera_z
    num_models = len(models)
    # Predict using all models
    new_out = np.zeros((np.shape(new_input)[0], num_models, 16))
    avg_preds = []
    for i in range(np.shape(new_input)[0]):
        print('Input going into the NN', new_input[i])
        for j in range(num_models):
            new_out[i, j] = nn_interpolation(new_input[i], min_in, max_in, models[j])
    avg_preds = np.mean(new_out, axis=1)

    # Average prediction is now the ratio:
    avg_preds_abs = avg_preds * pkz_fid  
    # Average prediction
    if rc == True:
        avg_preds_final = avg_preds_abs * bc_cdm * rc_cdm
    if rc == False:
        avg_preds_final = avg_preds_abs * bc_cdm
        
    return avg_preds_final

def get_nn_prediction_post_rs_mcmc(z, new_input, min_in, max_in, models, u042, u046, u05, Rs10_42, Rs10_46, Rs10_5):
    """
    Get predictions for a given input using single model obtained using train.py for the specified redshift.
    MCMCM is run with scaled T0 - so need to re-scale T0 back into original (in which NN is trained)
    Args:
        z (float): Redshift value.
        new_input (np.ndarray): Input data for prediction.
        min_in (np.ndarray): Minimum values of input features.
        max_in (np.ndarray): Maximum values of input features.
        models (list): List of trained models for the redshift.

    Returns:
        np.ndarray: Final predictions after applying BC correction.
    """

    new_input[0] = new_input[0] * (10**4)

    # Define a mapping of redshifts to BC correction file paths
    bc_cdm_paths = {
        4.2: "/data/og313b/lyalpha/models/cdm/planck1_20_1024/bc_4.2_1.0.txt",
        4.6: "/data/og313b/lyalpha/models/cdm/planck1_20_1024/bc_4.6_1.0.txt",
        5.0: "/data/og313b/lyalpha/models/cdm/planck1_20_1024/bc_5.0_1.0.txt"
    }

    spectrum_centers = np.arange(-2.2, -0.7, 0.1)


    # Ensure the redshift has a corresponding BC correction file

    if z not in bc_cdm_paths:
        raise ValueError(f"No BC correction file path defined for redshift {z}")

    # Get the BC correction file path for the redshift
    bc_cdm_path = bc_cdm_paths[z]
    bc_cdm = np.loadtxt(bc_cdm_path)


    Rs10z_interp1 = np.zeros((len(spectrum_centers)))

    # Perform linear interpolation
    for i in range(len(spectrum_centers)):

        if z == 4.2:
            for i in range(len(spectrum_centers)): #idx = 7 for fixing correction at tau_eff = 1
                Rs10z_interp1[i] = interp1d(u042, Rs10_42[:, 7, i], fill_value="extrapolate")(new_input[3])

        elif z == 4.6:
            for i in range(len(spectrum_centers)): #idx = 7 for fixing correction at tau_eff = 1
                Rs10z_interp1[i] = interp1d(u046, Rs10_46[:, 7, i], fill_value="extrapolate")(new_input[3])

        elif z == 5.0:
            for i in range(len(spectrum_centers)): #idx = 7 for fixing correction at tau_eff = 1
                Rs10z_interp1[i] = interp1d(u05, Rs10_5[:, 7, i], fill_value="extrapolate")(new_input[3])

    #Rs10z_interp1.reshape(-1) # 16 bins array including Rs for that u0 and tau_eff


    # Predict using all models
    num_models = len(models)
    new_out = np.zeros((num_models, 16))
    avg_preds = []
    for j in range(num_models):
        new_out[j] = nn_interpolation(new_input, min_in, max_in, models[j])
    avg_preds.append(np.mean(new_out, axis=0))
    # Average prediction

    avg_preds_final = avg_preds * bc_cdm * Rs10z_interp1

    return avg_preds_final



def get_nn_prediction_post_rs(z, new_input, min_in, max_in, models, u042, u046, u05, Rs10_42, Rs10_46, Rs10_5):
    """
    Get predictions for a given input using single model obtained using train.py for the specified redshift.
    MCMCM is run with scaled T0 - so need to re-scale T0 back into original (in which NN is trained)
    Args:
        z (float): Redshift value.
        new_input (np.ndarray): Input data for prediction.
        min_in (np.ndarray): Minimum values of input features.
        max_in (np.ndarray): Maximum values of input features.
        models (list): List of trained models for the redshift.

    Returns:
        np.ndarray: Final predictions after applying BC correction.
    """

    #new_input[0] = new_input[0] * (10**4)

    # Define a mapping of redshifts to BC correction file paths
    bc_cdm_paths = {
        4.2: "/data/og313b/lyalpha/models/cdm/planck1_20_1024/bc_4.2_1.0.txt",
        4.6: "/data/og313b/lyalpha/models/cdm/planck1_20_1024/bc_4.6_1.0.txt",
        5.0: "/data/og313b/lyalpha/models/cdm/planck1_20_1024/bc_5.0_1.0.txt"
    }

    spectrum_centers = np.arange(-2.2, -0.7, 0.1)


    # Ensure the redshift has a corresponding BC correction file

    if z not in bc_cdm_paths:
        raise ValueError(f"No BC correction file path defined for redshift {z}")

    # Get the BC correction file path for the redshift
    bc_cdm_path = bc_cdm_paths[z]
    bc_cdm = np.loadtxt(bc_cdm_path)


    num_models = len(models)
    new_out = np.zeros((np.shape(new_input)[0], num_models, 16))
    avg_preds = []

    for i in range(np.shape(new_input)[0]):
        Rs10z_interp1 = np.zeros((len(spectrum_centers)))
    
        # Perform linear interpolation for each array in new_input
        for j in range(len(spectrum_centers)):
            if z == 4.2:
                Rs10z_interp1[j] = interp1d(u042, Rs10_42[:, 7, j], fill_value="extrapolate")(new_input[i, 3])
            elif z == 4.6:
                Rs10z_interp1[j] = interp1d(u046, Rs10_46[:, 7, j], fill_value="extrapolate")(new_input[i, 3])
            elif z == 5.0:
                Rs10z_interp1[j] = interp1d(u05, Rs10_5[:, 7, j], fill_value="extrapolate")(new_input[i, 3])
    
        print('Input going into the NN', new_input[i])
        for j in range(num_models):
            new_out[i, j] = nn_interpolation(new_input[i], min_in, max_in, models[j])
    
        avg_preds.append(np.mean(new_out[i], axis=0) * bc_cdm * Rs10z_interp1)

    avg_preds_final = np.array(avg_preds)

    return avg_preds_final
