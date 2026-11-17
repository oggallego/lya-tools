#%%
## train.py

from data_processing import load_preprocessed_data
from model import NNInterpolator
from utils.plotting import plot_loss_vs_epoch

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import time

def train_model(config, z):
    """
    Train the model using preprocessed data and save results.

    Args:
        config (dict): Configuration dictionary for the redshift.
        z (float): Redshift value.
    """
    # Initialize the model
    model = NNInterpolator(
        input_size=6,
        output_size=16,
        hidden=config["hidden"],
        learning_rate=config["learning_rate"],
        act=torch.nn.ReLU,
        patience=config["patience"],
        patience_scheduler=config["patience_scheduler"],
        base_path=config["base_path"]
    )

    # Load preprocessed data using the model's output path logic
    inputs, outputs = load_preprocessed_data(model, z)

    # Split data into training and validation sets
    train_idx, test_idx = train_test_split(range(len(inputs)), test_size=0.1, random_state=42)
    train_input, val_input = inputs[train_idx], inputs[test_idx]
    train_output, val_output = outputs[train_idx], outputs[test_idx]

    print(np.shape(train_input, train_output))
    print(np.shape(val_input), np.shape(val_output))

    # Convert to PyTorch tensors
    train_input, train_output = torch.from_numpy(train_input).float(), torch.from_numpy(train_output).float()
    val_input, val_output = torch.from_numpy(val_input).float(), torch.from_numpy(val_output).float()

    # Create DataLoaders
    train_data = TensorDataset(train_input, train_output)
    val_data = TensorDataset(val_input, val_output)
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=False)

    # Train the model
    start = time.time()
    train_losses, val_losses, y_true, y_pred, best_epoch = model.train_batch(
        z, train_loader, val_loader, config["epochs"], fold = 'train'
    )
    end = time.time()
    print(f"Training time: {end - start} seconds")

    # Save the loss plot
    output_path = model.get_output_path(z)
    plot_loss_vs_epoch(train_losses, val_losses, f"{output_path}/loss_vs_epoch.pdf")


def check_test_distribution(model, z, base_path, fold):
    """
    Inspect the test distribution after training.

    Args:
        model (NNInterpolator): Trained model instance.
        z (float): Redshift.
        base_path (str): Path to the directory containing test data and model.
        fold (str): Fold identifier (e.g., 'train').
    """
    # Load test data
    test_data_path = f"{base_path}/z{z}/test_data.npz"
    print(f'Reading data in base_path {test_data_path}')
    test_data = np.load(test_data_path)

    # Split test data into inputs and outputs
    test_x = test_data['x']
    test_y = test_data['y']

    # Convert to tensors
    test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
    test_y_tensor = torch.tensor(test_y, dtype=torch.float32)

    # Load the model's saved state
    model_path = f"{base_path}/z{z}/last_model_foldno_{fold}_120x3.pth"
    print('Reading the model in', model_path)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.model.load_state_dict(state_dict)
    model.model.eval()

    # Predict on the test set
    with torch.no_grad():
        test_predictions = model.model(test_x_tensor)

    # Convert predictions and true values to numpy arrays
    test_predictions = test_predictions.numpy()
    print(test_predictions)
    test_output = test_y_tensor.numpy()
    print(test_output)

    # Compute the ratio of predictions to true values
    ratio = test_predictions / test_output
    print('Predictions',np.exp(test_predictions) )
    print('True', np.exp(test_output))
    print('Loss', (((np.mean(test_predictions, axis = 1) - np.mean(test_output, axis = 1)))**2)[-8:])
    print('Loss', (((np.mean(test_predictions, axis = 1) - np.mean(test_output, axis = 1)))**2)[:8])
    median = np.percentile(ratio, 50, axis=0)
    lower_68 = np.percentile(ratio, 16, axis=0)
    upper_68 = np.percentile(ratio, 84, axis=0)
    lower_95 = np.percentile(ratio, 2.5, axis=0)
    upper_95 = np.percentile(ratio, 97.5, axis=0)

    # Print metrics
    print(f"Median: {median}")
    print(f"68% confidence interval: {lower_68} - {upper_68}")
    print(f"95% confidence interval: {lower_95} - {upper_95}")

    one_sigma_error = upper_68 - lower_68
    print(f"One sigma error: {one_sigma_error}")
    

    # Generate the plot
    logk = np.arange(-2.2, -0.7, 0.1)  # Example range of log(k)
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 21})

    # Median line
    plt.plot(logk, median, color='mediumblue', linewidth=2, label="Median")

    # Confidence intervals
    plt.fill_between(logk, lower_68, upper_68, color='blue', alpha=0.5)#, label="68% CI")
    plt.fill_between(logk, lower_95, upper_95, color='cornflowerblue', alpha=0.5)#, label="95% CI")

    # Reference lines
    plt.axhline(y=1.01, color='dimgray', linestyle='--')
    plt.axhline(y=0.99, color='dimgray', linestyle='--')
    plt.axhline(y=1.00, color='darkslategray', linestyle='-')

    # Formatting
    plt.xlim(-2.2, -0.7)
    plt.xticks([-2.2, -1.5, -1.0])
    plt.ylabel('Log(Predicted)/Log(True)')
    plt.xlabel('$\mathrm{log}_{10}$ ($k$ $\mathrm{[km^{-1}\,s]})$')
    plt.legend()
    plt.tight_layout()

    # Save and show the plot
        # Save the loss plot
    output_path = model.get_output_path(z)
    print(f'File saved in {output_path}/kcross_train_fold_120x3_test.pdf')
    output_file = f"{output_path}/kcross_train_fold_120x3_test.pdf"
    print('Output path', output_path)
    plt.savefig(output_file)
    plt.show()




def check_test_distribution_10k(model, z, base_path, num_folds, fold):
    from sklearn.metrics import mean_squared_error
    """
    Inspect the test distribution after training.

    Args:
        model (NNInterpolator): Trained model instance.
        z (float): Redshift.
        base_path (str): Path to the directory containing test data and model.
        fold (str): Fold identifier (e.g., 'train').
    """
    # Load test data
    test_data_path = f"{base_path}/z{z}/test_data.npz"
    print(f'Reading data in base_path {test_data_path}')
    test_data = np.load(test_data_path)

    # Split test data into inputs and outputs
    test_x = test_data['x']
    test_y = test_data['y']

    # Convert to tensors
    test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
    test_y_tensor = torch.tensor(test_y, dtype=torch.float32)

    # Number of folds
    # num_folds = 10  # Replace with your actual number of folds

    # Initialize a dictionary to store the sum of weights
    sum_state_dict = None
    fracerr = []
    for fold in range(num_folds):
        # Load the state dictionary from each fold
        state_dict = torch.load(f'{base_path}/z{z}/last_model_foldno_{fold}_120x3.pth', map_location=torch.device('cpu'))
        # Modify the keys to match the model's keys
        model.model.load_state_dict(state_dict)
        model.model.eval()

        # Make predictions
        with torch.no_grad():
            test_predictions = model.model(test_x_tensor)

        # Convert the tensor of predictions to a numpy array
        test_predictions = test_predictions.detach().numpy()
        test_output = test_y_tensor.detach().numpy()


        # Calculate the MSE of the predictions
        mse = mean_squared_error(test_output, test_predictions)
        print(f'Mean Squared Error for fold {fold}: {mse}')

        fracerr.append(np.exp(test_predictions)/np.exp(test_output))

    print(np.shape(fracerr))
    mean_fracerr = np.mean(np.array(fracerr), axis=0)
    ratio = mean_fracerr
    median = np.percentile(ratio, 50, axis=0)
    lower_68 = np.percentile(ratio, 16, axis=0)
    upper_68 = np.percentile(ratio, 84, axis=0)
    lower_95 = np.percentile(ratio, 2.5, axis=0)
    upper_95 = np.percentile(ratio, 97.5, axis=0)

    print(mean_fracerr)
    print(lower_68)
    print(upper_68)
    print(lower_95)
    print(upper_95)
    one_sigma_error = upper_68 - lower_68
    print(one_sigma_error)

    # Define figure and main axis
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()  # Create the secondary y-axis

    # Plot the median line
    logk = np.arange(-2.2, -0.7, 0.1)
    plt.rcParams.update({'font.size': 20})

    ax1.axhline(y=0.99, color='black', linestyle='--')

    # Plot the 68% and 95% confidence intervals
    ax1.fill_between(logk, lower_68, upper_68, color='blue', alpha=0.5)
    ax1.fill_between(logk, lower_95, upper_95, color='cornflowerblue', alpha=0.5)

    # Add dashed lines at 1.01 and 0.99
    ax1.axhline(y=1.01, color='black', linestyle='--')
    ax1.axhline(y=1.00, color='darkslategray')
    ax1.plot(logk, median, color='mediumblue', linewidth=2)

    # Set consistent y-axis limits
    ax1.set_ylim(0.9875, 1.025)
    ax2.set_ylim(0.985, 1.025)  # Ensure twin axis matches
    ax1.set_xlim(-2.2, -0.7)

    # Define matching tick marks
    yticks = np.linspace(0.98, 1.02, 5)  # Adjust number of ticks
    ax1.set_yticks(yticks)
    ax2.set_yticks(yticks)
    ax2.set_yticklabels([f'{100 * (y - 1):.1f}%' for y in yticks])

    # Labels
    ax1.set_ylabel('Predicted/True')
    ax1.set_xlabel('$\mathrm{log}_{10}$ ($k$ $\mathrm{[km^{-1}\,s])}$')
    ax2.set_ylabel('Percentage')

    # Add text
    ax1.text(-1.2, 1.02, 'Molaro+2023', fontsize=14, color='dimgray')
    ax1.axhline(y=1.25, color='dimgray', linestyle=':')
    ax1.axhline(y=0.972, color='dimgray', linestyle=':')

    # Save and show
    plt.savefig('kcross_test_120x3_10folds_paper.pdf', bbox_inches='tight')
    plt.show()

# def check_test_distribution_10k_allz(model, base_path):
#     from sklearn.metrics import mean_squared_error
#     """
#     Inspect the test distribution after training.

#     Args:
#         model (NNInterpolator): Trained model instance.
#         z (float): Redshift.
#         base_path (str): Path to the directory containing test data and model.
#         fold (str): Fold identifier (e.g., 'train').
#     """

#     import numpy as np
#     import torch
#     import matplotlib.pyplot as plt
#     from sklearn.metrics import mean_squared_error

#     # Define figure with 3 columns, sharing y-axis
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
#     plt.rcParams.update({'font.size': 20})

#     z_values = [4.2, 4.6, 5.0]  # Update with your actual z values

#     for idx, z in enumerate(z_values):
#         test_data_path = f"{base_path}/z{z}/test_data.npz"
#         print(f'Reading data in base_path {test_data_path}')
#         test_data = np.load(test_data_path)

#         test_x = test_data['x']
#         test_y = test_data['y']

#         test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
#         test_y_tensor = torch.tensor(test_y, dtype=torch.float32)

#         if z == 4.2:
#             num_folds = 1
#         elif z == 4.6:
#             num_folds = 1
#         elif z == 5.0:
#             num_folds = 1

#         fracerr = []
#         for fold in range(num_folds):
#             state_dict = torch.load(f'{base_path}/z{z}/last_model_foldno_{fold}_120x3.pth', map_location=torch.device('cpu'))
#             model.model.load_state_dict(state_dict)
#             model.model.eval()

#             with torch.no_grad():
#                 test_predictions = model.model(test_x_tensor)

#             test_predictions = test_predictions.detach().numpy()
#             test_output = test_y_tensor.detach().numpy()

#             mse = mean_squared_error(test_output, test_predictions)
#             print(f'Mean Squared Error for fold {fold}: {mse}')

#             fracerr.append(np.exp(test_predictions) / np.exp(test_output))

#         ratio = np.mean(np.array(fracerr), axis=0)
#         median = np.percentile(ratio, 50, axis=0)
#         lower_68 = np.percentile(ratio, 16, axis=0)
#         upper_68 = np.percentile(ratio, 84, axis=0)

#         lower_95 = np.percentile(ratio, 2.5, axis=0)
#         upper_95 = np.percentile(ratio, 97.5, axis=0)
#         print(lower_95, upper_95)
#         logk = np.arange(-2.2, -0.7, 0.1)

#         # Plot on the corresponding subplot
#         ax = axes[idx]
#         ax.tick_params(axis='both', labelsize=20)
#         ax.axhline(y=1, color='black', linestyle='--')
#         # ax.axhline(y=1.01, color='black', linestyle='--')
        
#         ax.fill_between(logk, lower_68, upper_68, color='blue', alpha=0.5)
#         ax.fill_between(logk, lower_95, upper_95, color='cornflowerblue', alpha=0.5)
#         print('Lower 95 should be plotted')
#         ax.plot(logk, median, color='mediumblue', linewidth=2)

#         # Gray dashed lines
#         # if idx == 0:
#         #     ax.axhline(y=1.025, color='dimgray', linestyle=':')
#         #     ax.axhline(y=0.972, color='dimgray', linestyle=':')

#         # if idx == 1:
#         #     ax.axhline(y=1.027, color='dimgray', linestyle=':')
#         #     ax.axhline(y=0.978, color='dimgray', linestyle=':')
        
#         # if idx == 2:
#         #     ax.axhline(y=1.029, color='dimgray', linestyle=':')
#         #     ax.axhline(y=0.974, color='dimgray', linestyle=':')

#         # Set limits
#         ax.set_xlim(-2.2, -0.7)
#         ax.set_ylim(0.99, 1.01)

#         # Add text in the middle subplot
#         #if idx == 1:
#         #    ax.text(-1.4, 1.02, 'Molaro+2023', fontsize=20, color='dimgray')

#         # Only create ax2 (percentage axis) for the last column
#         #if idx == 2:
#         #    ax2 = ax.twinx()
#         #    ax2.set_ylim(ax.get_ylim())  # Ensure exact match with ax

#             # Use the same tick positions as ax
#         #    yticks = ax.get_yticks()
#         #    ax2.set_yticks(yticks)

#             # Convert y-values to percentage relative to 1
#                 # Explicitly set both ticks and labels
#         #    percent_ticks = [100 * (y - 1) for y in yticks]
#         #    ax2.set_yticks(yticks)  
#         #    ax2.set_yticklabels([f'{p:.1f}%' for p in percent_ticks])
#         #    ax2.set_yticklabels([f'{100 * (y - 1):.1f}%' for y in yticks])

#         #    ax2.set_ylabel('Percentage')
#         #    print(f'Y-ticks (left axis): {yticks}')
#         #    print(f'Converted percentages: {[100 * (y - 1) for y in yticks]}')

#         # Labels and title
#         #ax.set_title(f'z = {z}')
#         #ax.set_xlabel('$\mathrm{log}_{10}$ ($k$ $\mathrm{[km^{-1}\,s])}$', fontsize = 20)

#     # Shared y-axis label
#     axes[0].set_ylabel('Predicted/True', fontsize = 20)

#     ####### ON TOP OF THIS PLOT THE RESULT FOR THE 10 FOLDS:
  
#     for idx, z in enumerate(z_values):
#             test_data_path = f"{base_path}/z{z}/test_data.npz"
#             print(f'Reading data in base_path {test_data_path}')
#             test_data = np.load(test_data_path)

#             test_x = test_data['x']
#             test_y = test_data['y']

#             test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
#             test_y_tensor = torch.tensor(test_y, dtype=torch.float32)

#             if z == 4.2:
#                 num_folds = 10
#             elif z == 4.6:
#                 num_folds = 10
#             elif z == 5.0:
#                 num_folds = 10

#             fracerr = []
#             for fold in range(num_folds):
#                 state_dict = torch.load(f'{base_path}/z{z}/last_model_foldno_{fold}_120x3.pth', map_location=torch.device('cpu'))
#                 model.model.load_state_dict(state_dict)
#                 model.model.eval()

#                 with torch.no_grad():
#                     test_predictions = model.model(test_x_tensor)

#                 test_predictions = test_predictions.detach().numpy()
#                 test_output = test_y_tensor.detach().numpy()

#                 mse = mean_squared_error(test_output, test_predictions)
#                 print(f'Mean Squared Error for fold {fold}: {mse}')

#                 fracerr.append(np.exp(test_predictions) / np.exp(test_output))

#             ratio = np.mean(np.array(fracerr), axis=0)
#             median = np.percentile(ratio, 50, axis=0)
#             lower_68 = np.percentile(ratio, 16, axis=0)
#             upper_68 = np.percentile(ratio, 84, axis=0)

#             lower_95 = np.percentile(ratio, 2.5, axis=0)
#             upper_95 = np.percentile(ratio, 97.5, axis=0)
#             print(lower_95, upper_95)
#             logk = np.arange(-2.2, -0.7, 0.1)

#             # Plot on the corresponding subplot
#             ax = axes[idx]
#             ax.tick_params(axis='both', labelsize=20)
#             ax.axhline(y=1, color='black', linestyle='--')
#             # ax.axhline(y=1.01, color='black', linestyle='--')
            
#             ax.fill_between(logk, lower_68, upper_68, color='red', alpha=0.5)
#             ax.fill_between(logk, lower_95, upper_95, color='tomato', alpha=0.5)
#             print('Lower 95 should be plotted')
#             ax.plot(logk, median, color='brown', linewidth=2)

#     # Save and show figure
#     plt.tight_layout()
#     plt.savefig('comparison_3columns_NEW_all.pdf', bbox_inches='tight')
#     plt.show()






    # Load test data
    # for z in z_values:
    #     test_data_path = f"{base_path}/z{z}/test_data.npz"
    #     print(f'Reading data in base_path {test_data_path}')
    #     test_data = np.load(test_data_path)

    #     # Split test data into inputs and outputs
    #     test_x = test_data['x']
    #     test_y = test_data['y']

    #     # Convert to tensors
    #     test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
    #     test_y_tensor = torch.tensor(test_y, dtype=torch.float32)

    #     # Number of folds
    #     # num_folds = 10  # Replace with your actual number of folds

    #     # Initialize a dictionary to store the sum of weights
    #     sum_state_dict = None

    #     fracerr = []
    #     if z == 4.2:
    #         num_folds = 10
    #     if z == 4.6:
    #         num_folds == 5
    #     if z == 5.0:
    #         num_folds = 1
    #     for fold in range(num_folds):
    #         # Load the state dictionary from each fold
    #         state_dict = torch.load(f'{base_path}/z{z}/last_model_foldno_{fold}_120x3.pth', map_location=torch.device('cpu'))
    #         # Modify the keys to match the model's keys
    #         model.model.load_state_dict(state_dict)
    #         model.model.eval()

    #         # Make predictions
    #         with torch.no_grad():
    #             test_predictions = model.model(test_x_tensor)

    #         # Convert the tensor of predictions to a numpy array
    #         test_predictions = test_predictions.detach().numpy()
    #         test_output = test_y_tensor.detach().numpy()


    #         # Calculate the MSE of the predictions
    #         mse = mean_squared_error(test_output, test_predictions)
    #         print(f'Mean Squared Error for fold {fold}: {mse}')

    #         fracerr.append(np.exp(test_predictions)/np.exp(test_output))

    #     print(np.shape(fracerr))
    #     mean_fracerr = np.mean(np.array(fracerr), axis=0)
    #     ratio = mean_fracerr
    #     median = np.percentile(ratio, 50, axis=0)
    #     lower_68 = np.percentile(ratio, 16, axis=0)
    #     upper_68 = np.percentile(ratio, 84, axis=0)
    #     lower_95 = np.percentile(ratio, 2.5, axis=0)
    #     upper_95 = np.percentile(ratio, 97.5, axis=0)

    #     print(mean_fracerr)
    #     print(lower_68)
    #     print(upper_68)
    #     print(lower_95)
    #     print(upper_95)
    #     one_sigma_error = upper_68 - lower_68
    #     print(one_sigma_error)

    # # Define figure and main axis
    # fig, ax1 = plt.subplots(figsize=(8, 6))
    # ax2 = ax1.twinx()  # Create the secondary y-axis

    # # Plot the median line
    # logk = np.arange(-2.2, -0.7, 0.1)
    # plt.rcParams.update({'font.size': 20})

    # ax1.axhline(y=0.99, color='black', linestyle='--')

    # # Plot the 68% and 95% confidence intervals
    # ax1.fill_between(logk, lower_68, upper_68, color='blue', alpha=0.5)
    # ax1.fill_between(logk, lower_95, upper_95, color='cornflowerblue', alpha=0.5)

    # # Add dashed lines at 1.01 and 0.99
    # ax1.axhline(y=1.01, color='black', linestyle='--')
    # ax1.axhline(y=1.00, color='darkslategray')
    # ax1.plot(logk, median, color='mediumblue', linewidth=2)

    # # Set consistent y-axis limits
    # ax1.set_ylim(0.985, 1.025)
    # ax2.set_ylim(0.985, 1.025)  # Ensure twin axis matches
    # ax1.set_xlim(-2.2, -0.7)

    # # Define matching tick marks
    # yticks = np.linspace(0.98, 1.02, 5)  # Adjust number of ticks
    # ax1.set_yticks(yticks)
    # ax2.set_yticks(yticks)
    # ax2.set_yticklabels([f'{100 * (y - 1):.1f}%' for y in yticks])

    # # Labels
    # ax1.set_ylabel('Predicted/True')
    # ax1.set_xlabel('$\mathrm{log}_{10}$ ($k$ $\mathrm{[km^{-1}\,s])}$')
    # ax2.set_ylabel('Percentage')

    # # Add text
    # ax1.text(-1.2, 1.02, 'Molaro+2023', fontsize=14, color='dimgray')
    # ax1.axhline(y=1.25, color='dimgray', linestyle=':')
    # ax1.axhline(y=0.972, color='dimgray', linestyle=':')

    # # Save and show
    # plt.savefig('kcross_test_120x3_10folds_paper.pdf', bbox_inches='tight')
    # plt.show()



def check_test_distribution_10k_allz(model, base_path):
    from sklearn.metrics import mean_squared_error
    """
    Inspect the test distribution after training.

    Args:
        model (NNInterpolator): Trained model instance.
        z (float): Redshift.
        base_path (str): Path to the directory containing test data and model.
        fold (str): Fold identifier (e.g., 'train').
    """

    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error


    # Define figure with 3 columns, sharing y-axis
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    plt.rcParams.update({'font.size': 20})

    z_values = [4.2, 4.6, 5.0]  # Update with your actual z values

    for idx, z in enumerate(z_values):
        base_path = '/data/og313/lyalpha/code_wdm/analysis_no_RC_kfold'
        test_data_path = f"{base_path}/z{z}/test_data.npz"
        print(f'Reading data in base_path {test_data_path}')
        test_data = np.load(test_data_path)

        test_x = test_data['x']
        test_y = test_data['y']

        test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
        test_y_tensor = torch.tensor(test_y, dtype=torch.float32)

        # ------ PLOT FOR 1 FOLD ------
        num_folds = 1
        fracerr = []

        fold = 4
        state_dict = torch.load(f'{base_path}/z{z}/last_model_foldno_{fold}_120x3.pth', map_location=torch.device('cpu'))
        model.model.load_state_dict(state_dict)
        model.model.eval()

        with torch.no_grad():
            test_predictions = model.model(test_x_tensor)

        test_predictions = test_predictions.detach().numpy()
        test_output = test_y_tensor.detach().numpy()

        mse = mean_squared_error(test_output, test_predictions)
        #print(f'Mean Squared Error for fold {fold}: {mse}')

        fracerr.append(np.exp(test_predictions) / np.exp(test_output))

        ratio = np.mean(np.array(fracerr), axis=0)
        median = np.percentile(ratio, 50, axis=0)
        lower_68 = np.percentile(ratio, 16, axis=0)
        upper_68 = np.percentile(ratio, 84, axis=0)
        lower_95 = np.percentile(ratio, 2.5, axis=0)
        upper_95 = np.percentile(ratio, 97.5, axis=0)

        logk = np.arange(-2.2, -0.7, 0.1)
        ax = axes[idx]
        ax.tick_params(axis='both', labelsize=20)
        ax.axhline(y=1, color='black', linestyle='--')

        # 1-Fold Results (Blue Shades)
        ax.fill_between(logk, lower_68, upper_68, color='blue', alpha=0.3)  # Increased transparency
        ax.fill_between(logk, lower_95, upper_95, color='cornflowerblue', alpha=0.3)
        ax.plot(logk, median, color='mediumblue', linewidth=2, label=f"1 fold")

        # ------ PLOT FOR 10 FOLDS ------
        num_folds = 10
        fracerr = []
        for fold in range(num_folds):
            state_dict = torch.load(f'{base_path}/z{z}/last_model_foldno_{fold}_120x3.pth', map_location=torch.device('cpu'))
            model.model.load_state_dict(state_dict)
            model.model.eval()

            with torch.no_grad():
                test_predictions = model.model(test_x_tensor)

            test_predictions = test_predictions.detach().numpy()
            test_output = test_y_tensor.detach().numpy()

            mse = mean_squared_error(test_output, test_predictions)
            print(f'Mean Squared Error for fold {fold}: {mse}')

            fracerr.append(np.exp(test_predictions) / np.exp(test_output))

        ratio = np.mean(np.array(fracerr), axis=0)
        median = np.percentile(ratio, 50, axis=0)
        lower_68 = np.percentile(ratio, 16, axis=0)
        upper_68 = np.percentile(ratio, 84, axis=0)
        lower_95 = np.percentile(ratio, 2.5, axis=0)
        upper_95 = np.percentile(ratio, 97.5, axis=0)

        # 10-Fold Results (Red Shades)
        ax.fill_between(logk, lower_68, upper_68, color='red', alpha=0.4)  # Increased transparency
        ax.fill_between(logk, lower_95, upper_95, color='tomato', alpha=0.4)
        ax.plot(logk, median, color='brown', linewidth=2, linestyle='-', label="10 folds")
        #ax.axhline(y=1.01, color='black', linestyle='--')
        #ax.axhline(y=0.99, color='black', linestyle='--')

        # Set limits
        ax.set_xlim(-2.2, -0.7)
        ax.set_ylim(0.99, 1.01)
        ax.set_title(f'z = {z}')
        ax.set_xlabel('$\mathrm{log}_{10}$ ($k$ $\mathrm{[km^{-1}\,s])}$', fontsize = 20)

        if idx == 2:
           ax2 = ax.twinx()
           ax2.set_ylim(ax.get_ylim())  # Ensure exact match with ax

            # Use the same tick positions as ax
           yticks = ax.get_yticks()
           ax2.set_yticks(yticks)

            # Convert y-values to percentage relative to 1
            #     Explicitly set both ticks and labels
           percent_ticks = [100 * (y - 1) for y in yticks]
           ax2.set_yticks(yticks)  
           ax2.set_yticklabels([f'{p:.1f}%' for p in percent_ticks])
           ax2.set_yticklabels([f'{100 * (y - 1):.1f}%' for y in yticks])

           ax2.set_ylabel('Percentage')
           print(f'Y-ticks (left axis): {yticks}')
           print(f'Converted percentages: {[100 * (y - 1) for y in yticks]}')

    # Shared y-axis label
    axes[0].set_ylabel('Predicted/True', fontsize=20)

    # Add legend to the last subplot
    axes[-1].legend(loc='upper left', fontsize=18, frameon=False)

    # Save and show figure
    plt.tight_layout()
    plt.savefig('comparison_3columns_NEW_all.pdf', bbox_inches='tight')
    plt.show()


# %%
