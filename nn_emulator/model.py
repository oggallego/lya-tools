# model.py
import json

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchviz import make_dot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np 
import os
import re
import ast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import redshift_configs
import time
import platform
import json
import torch
import psutil


class NNInterpolator:
    def __init__(self, input_size, output_size, hidden, learning_rate, act, patience, patience_scheduler, base_path):
        """
        Initialize the NNInterpolator.
        Set up imitates Molaros' code.

        """
        # torch.manual_seed(seed)
        # np.random.seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden),
            act(),
            nn.Linear(hidden, hidden),
            act(),
            nn.Linear(hidden, hidden),
            act(),
            nn.Linear(hidden, hidden),
            act(),
            nn.Linear(hidden, output_size),

        )
        self.init_weights()
        self.model = self.model.to(self.device)
        self.learning_rate = learning_rate # Original value passed to the model
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.patience_scheduler = patience_scheduler
      
        #self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.1)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience = self.patience_scheduler, min_lr = 1e-6)
        self.loss_fn = nn.MSELoss()
        self.patience = patience
        self.base_path = base_path
        

    # Add a visualization method:
    def visualize(self):
        # Correct input shape is (1, 5), matching the input size of the first layer
        x = torch.randn(1, 5).to(self.device)  # Input size should match the actual input size (5)
        output = self.model(x)
        dot = make_dot(output, params=dict(self.model.named_parameters()))
        dot.render("NNInterpolator_diagram", format="png")
        return dot

    def init_weights(self, seed=42):
        def init_layer(layer):
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight, gain=1.0)
                torch.nn.init.zeros_(layer.bias)

        self.model.apply(init_layer)

    def convert_redshift(self, z):
        if z.is_integer():
            return int(z)
        else:
            return int(str(z).replace('.', ''))
        
    def get_hardware_info(self):
        return {
            "device": str(self.device),
            "cpu_model": platform.processor(), #returns the processor architecture                                                                                                            
            "cpu_cores_physical": psutil.cpu_count(logical=False), #the number of real hardware cores                                                                                         
            "cpu_cores_logical": psutil.cpu_count(logical=True),                                                                                                                             
            "ram_gb": round(psutil.virtual_memory().total / 1e9, 2), #convert to GB                                                                                                           
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if torch.cuda.is_available() else None,
    }

    def preprocess_data(self, z, inputz, psz, batch_size):

        np.random.seed(42)
        # Random split of the data:
        indices = np.arange(inputz.shape[0])
        np.random.shuffle(indices)

        inputz = inputz[indices]
        psz = psz[indices]

        # Calculate the sizes of the splits
        train_size = int(0.8 * inputz.shape[0])  
        val_size = int(0.1 * inputz.shape[0])  
        test_size = inputz.shape[0] - train_size - val_size  # 

        # Split the input features
        train_input = inputz[:train_size]
        val_input = inputz[train_size:train_size+val_size]
        test_input = inputz[train_size+val_size:]
        #test_input = inputz[train_size:]

        # Split the output features
        train_output = psz[:train_size]
        val_output =psz[train_size:train_size+val_size]
        test_output = psz[train_size+val_size:]
        #test_output = psz[train_size:]

        # CHeck the distribution of the train output:
        # historgram
        import matplotlib.pyplot as plt
        plt.hist(train_output.flatten(), bins=50, alpha=0.5, label='Train Output')
        plt.xlabel('Pk')    #')
        plt.ylabel('Frequency')
        plt.title('Distribution of Train Output')
        plt.legend()
        plt.show()


        # Normalize training input data using min-max normalization:

        # Calculate min and max values from the training data
        min_in = np.min(train_input, axis=0)
        max_in = np.max(train_input, axis=0)

        #min_out = np.min(train_output, axis=0)
        #max_out = np.max(train_output, axis=0)

        # Apply min-max normalization to the training data
        train_x_normalized = (train_input - min_in) / (max_in - min_in)
        val_x_normalized = (val_input - min_in) / (max_in - min_in)
        test_x_normalized = (test_input - min_in) / (max_in - min_in)

        train_x_tensor = torch.tensor(train_x_normalized, dtype=torch.float32).to(self.device)
        val_x_tensor = torch.tensor(val_x_normalized, dtype=torch.float32).to(self.device)
        test_x_tensor  = torch.tensor(test_x_normalized, dtype=torch.float32).to(self.device)
        train_y_tensor = torch.tensor(train_output, dtype=torch.float32).to(self.device)
        val_y_tensor = torch.tensor(val_output, dtype=torch.float32).to(self.device)
        test_y_tensor  = torch.tensor(test_output, dtype=torch.float32).to(self.device)

    
        print(f"train_input is on device: {train_x_tensor.device}")
        #print(f"train_output is on device: {train_output.device}")                                                                                                                           
        print(f"val_input is on device: {test_y_tensor.device}")
        print(f"val_output is on device: {val_y_tensor.device}") 

        # 2. Create the PyTorch Dataset using the GPU tensors                                                                                                                                 
        train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_y_tensor)
        val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_y_tensor)
        test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        #For testing if wanting to reproduce the results then no need for reshuffle:
        
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


        # Save training, validation, and test data in the corresponding redshift directory:
        directory_path = f'z{self.convert_redshift(z)}/'
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        output_path = self.get_output_path(z)

        # Create the directory if it does not exist
        print('Save training and test data at!', output_path)
        np.savez(f'{output_path}/train_data.npz', x=train_x_normalized, y=train_output)
        # np.savez(f'{output_path}/val_data.npz', x=val_x_normalized, y=val_output)
        np.savez(f'{output_path}/val_data.npz', x=val_x_normalized, y=val_output)
        np.savez(f'{output_path}/test_data.npz', x=test_x_normalized, y=test_output)

        return train_loader,val_loader, min_in, max_in


    def train_batch(self, z, train_loader, val_loader, epochs, fold):
        # print(f"Number of epochs: {epochs}")
        # print(f"Length of val_loader: {len(val_loader)}")
        min_val_loss = np.inf
        best_model = None
        best_epoch = None
        counter = 0
        train_losses, val_losses = [], []
        all_y_true, all_y_pred = [], [] 
        update_lr = []

        ### INCLUDE SOME METRINCS TO BOOK-KEEP THE PERFORMANCE:
        config = redshift_configs[z]                                                                                                                               
        metrics = {
            "hardware": self.get_hardware_info(), #since it isa method on the class                                                                                                           
            "fold": fold,
            "z": z,
            "batch_size": None,
            "epoch_times_s": [],         # wall-clock seconds per epoch                                                                                                                       
            "train_losses": [],
            "min_epoch": config['min_epoch'],
            "val_losses": [],
            "total_wall_hours": None, #real time                                                                                                                                              
            "total_cpu_hours": None,
            "total_cpu_time_s": None, # actual CPU time (differs from wall if multiple threads)                                                                                               
            "effective_cores_used": None,
            "total_gpu_hours": None,
            "torch_num_threads": None,
            "epochs_completed": None,
            "early_stopped": False,
            "epochs_update_lr": None,
        }
        cpu_start = time.process_time()   # CPU time (not wall clock)                                                                                                                         
        wall_start = time.perf_counter()  # wall clock                                                                                                                                        


        print('Start timing when training')
        for epoch in range(epochs):
            
            epoch_start = time.perf_counter()
            self.model.train()
            train_loss = 0.0
            
            for x_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                # x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.model(x_batch.float())
                # # print(f"y_pred shape: {y_pred.shape}")
                loss = self.loss_fn(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_losses.append(train_loss / len(train_loader))


            self.model.eval()
            val_loss = 0
            y_true, y_pred_list = [], []
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    # x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    y_pred = self.model(x_batch)
                    val_loss += self.loss_fn(y_pred, y_batch).item()
                    y_true.append(y_batch.tolist())  
                    y_pred_list.append(y_pred.tolist())
            #         print(f"y_true shape before reshaping: {np.array(y_true).shape}")
            #         print(f"y_pred shape before reshaping: {np.array(y_pred_list).shape}")  
            # print(f"y_true shape before reshaping: {np.array(y_true).shape}")

            # print(f"y_pred shape before reshaping: {np.array(y_pred_list).shape}")
            num_output = 16
            # y_true = np.array(y_true).reshape(-1, num_output)
            # y_pred = np.array(y_pred_list).reshape(-1, num_output)
            y_true = np.vstack(y_true).reshape(-1, num_output)
            y_pred = np.vstack(y_pred_list).reshape(-1, num_output)

            val_losses.append(val_loss / len(val_loader))
    
            prev_lr = self.optimizer.param_groups[0]['lr'] #before updating lr
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']

            # Check updated learning rate:
            if new_lr < prev_lr:
                update_lr.append(epoch+1)

            prev_lr = new_lr


            ## SAVE SOME METRICS:
            metrics["train_losses"].append(train_losses[-1])
            metrics["val_losses"].append(val_losses[-1])

            metrics["epoch_times_s"].append(round(time.perf_counter() - epoch_start, 4)) #how long that epoch took     

            # print('Write in text file to load val and train losses before training finishes')
            np.savetxt(f"{output_path}/train_val_loss_bs{batch_size}_live.txt",np.column_stack((train_losses, val_losses)))  

            #EARLY STOPPING:                                                                                                                                                                  
            if val_losses[-1] < min_val_loss:
                min_val_loss = val_losses[-1]
                best_model = self.model.state_dict()
                best_epoch = epoch
                best_y_true, best_y_pred = y_true, y_pred
                counter = 0
            else:
                counter += 1

            print(f"Epoch {epoch+1}: Train Loss = {train_losses[-1]:.6f}, Val Loss = {val_losses[-1]:.6f}")

            # early stopping check                                                                                                                                                            
            if epoch > config['min_epoch'] and counter >= self.patience:
                print(f'Early stopping at epoch {epoch}')
                # epoch_train = epoch + 1
                metrics["early_stopped"] = True
                break
            # else:
            #     epoch_train = epoch + 1
        

        output_path = self.get_output_path(z)
        layers = [(m.in_features, m.out_features) for m in self.model if isinstance(m, nn.Linear)]
        n_hidden = layers[0][1]      # nodes per hidden layer                                                                                                                                 
        n_layers = len(layers) - 1   # exclude output layer                                                                                                                                   
        batch_size = config['batch_size']
        torch.save(self.model.state_dict(), f'{output_path}/model_{n_hidden}x{n_layers}_bs{batch_size}_ep{epochs}_{self.device}.pth')

        print(f"The best model was found at epoch {best_epoch + 1}")
        print('Finished Training')
        print('Finalize metrics')
        metrics["total_wall_hours"] = (round(time.perf_counter() - wall_start, 4))/3600
        metrics["total_cpu_time_s"] = round(time.process_time() - cpu_start, 4)
        metrics["total_cpu_hours"] = round(metrics["total_cpu_time_s"] / 3600, 6)
        metrics["batch_size"] = config['batch_size']
        metrics["effective_cores_used"] = round(metrics["total_cpu_time_s"] / (round(time.perf_counter() - wall_start, 4)), 1)

        metrics["total_gpu_hours"] = (round(metrics["total_wall_hours"] * torch.cuda.device_count(), 6) if self.device.type == 'cuda' else None)
        metrics["epochs_completed"]   = epoch + 1
        metrics["torch_num_threads"]  = torch.get_num_threads()
        metrics["best_epoch"]         = (best_epoch + 1) if best_epoch is not None else None
        metrics["best_val_loss"]      = min_val_loss
        metrics["epochs_update_lr"] = update_lr
        batch_size = config['batch_size']
        metrics_path = f"{output_path}/train_metrics_bs{batch_size}_ep{epoch}_{self.device}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")


        return train_losses, val_losses, best_y_true, best_y_pred, best_epoch, update_lr
    
    def convert_redshift(self, z):
        """
        Convert redshift to a string representation (if needed).
        """
        return f"{z:.1f}"
    
    def get_output_path(self, z):
        """
        Generate the output directory for a given redshift and create it if it does not exist.
        """
        output_path = os.path.join(self.base_path, f"z{self.convert_redshift(z)}")
        os.makedirs(output_path, exist_ok=True)
        return output_path
    
    def eval(self):
        self.model.eval() #use eval method from underlying PyTorch method

