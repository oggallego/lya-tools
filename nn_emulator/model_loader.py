import os
import torch
import numpy as np
from model import NNInterpolator
from data_processing import read_data_nn

def get_mixed_models(z, base_path):
    """
    Load trained models and associated data for a given redshift.

    Args:
        z (float): Redshift value.
        base_path (str): Base directory where models and preprocessed data are stored.

    Returns:
        tuple: (models, min_in, max_in, taueff)
            models (list): List of trained models for the redshift.
            min_in (np.ndarray): Minimum values for normalization.
            max_in (np.ndarray): Maximum values for normalization.
            taueff (float): Effective optical depth for the redshift.
    """
    learning_rate = 0.0001  # Fiducial value
    patience = 30
    hidden = 120
    act = torch.nn.ReLU
    models = []

    # Compute tau_eff for the redshift
    taueff = 1.56 * ((1 + z) / 5.75) ** 4
    
    print('Using base path', base_path) # This is defined in run cobaya script to use the current working directory
    inputs = np.loadtxt(f'{base_path}/z{z}/inputs{z}{base_path.split("analysis")[1]}_all.txt')
    print(f'Inputs are inputsz{base_path.split("analysis")[1]}_all.txt')
    print('Shape of all inputs loaded', np.shape(inputs))
    

    labels = np.loadtxt(f'{base_path}/z{z}/ps{z}{base_path.split("analysis")[1]}_all.txt')
    print('Shape of all inputs loaded', np.shape(labels))

    # Initialize the model
    model = NNInterpolator(
        input_size=inputs.shape[1],
        output_size=labels.shape[1],
        hidden=hidden,
        learning_rate=learning_rate,
        act=act,
        patience=patience,
        patience_scheduler=30,  # Not needed for inference
        base_path=base_path
    )

    # Preprocess data to get min_in and max_in for normalization
    _, _, min_in, max_in = model.preprocess_data(z, inputs, np.log(labels), batch_size=18)

    # Load models from saved states
    base_path_emulator = os.path.join(base_path, f"z{z}")
    print('Base path for emulator used is', base_path_emulator)
    model_files = [
        f for f in os.listdir(base_path_emulator)
        if f.startswith("last_model_foldno_") and f.endswith("_120x3.pth")
    ]
    print('Check model_files', model_files)

    if len(model_files) == 0:
        # Check for single-fold model
        single_model_path = os.path.join(base_path_emulator, "last_model_foldno_train_120x3.pth")
        if os.path.exists(single_model_path):
            print(f"Loading single model: {single_model_path}")
            model.model.load_state_dict(torch.load(single_model_path, map_location=torch.device('cpu')))
            model.model.eval()  # Set model to evaluation mode
            models.append(model)
        else:
            raise FileNotFoundError(f"No model files found for redshift {z} in {base_path_emulator}")
    else:
        # Load all fold models dynamically
        for model_file in model_files:
            model_path = os.path.join(base_path_emulator, model_file)
            print(f"Loading model: {model_path}")
            model.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.model.eval()  # Set model to evaluation mode
            models.append(model)

    return models, min_in, max_in, taueff

