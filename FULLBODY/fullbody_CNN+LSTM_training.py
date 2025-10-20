import os
import time
import gc
import sys
import copy
import math
import pickle
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.io
from scipy.signal import resample
from scipy.stats import pearsonr
from math import sqrt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import mne
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt

from models import create_model

print("All modules imported successfully!")

DATA_FOLDER="RightUpperLeg"
WINDOW_SIZE = 512  # 1 second of data (sfreq=1000Hz)
STEP_SIZE = int(0.5*(WINDOW_SIZE))     # 50% overlap between windows

# DataLoader Parameters
BATCH_SIZE = 64

# Train/Validation/Test Split Ratios (by trial for the selected subject)
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
TEST_RATIO=0.1
# TEST_RATIO is implicitly 1.0 - TRAIN_RATIO - VALIDATION_RATIO

for TARGET_SUBJECT_KEY in range(1,10):
    print(f"Loading pre-processed data for subject key: {TARGET_SUBJECT_KEY}...")
    try:
        # eeg_file_path = os.path.join(DATA_FOLDER, f"filtered_eeg_data_for_subject{TARGET_SUBJECT_KEY}")
        # with open(f"RightUpperLeg/filtered_eeg_data_for_subject{TARGET_SUBJECT_KEY}", 'rb') as f: # 'rb' means "read binary"
        with open(f"RightUpperLeg/filtered_eeg_data_for_subject{TARGET_SUBJECT_KEY}", 'rb') as f: # 'rb' means "read binary"
            eeg_subject_trials = pickle.load(f)


        kin_file_path = os.path.join(DATA_FOLDER, f"kin_data_for_subject{TARGET_SUBJECT_KEY}.npy")
        kin_subject_trials = np.load(kin_file_path, allow_pickle=True)
        
        print(f"Successfully loaded {len(eeg_subject_trials)} trials for the target subject.")

        # --- DEBUGGING PRINTS ---
        print("\n--- Data Loading Debug Info ---")
        print(f"Type of loaded EEG data object: {type(eeg_subject_trials)}")
        if isinstance(eeg_subject_trials, np.ndarray):
            print(f"Shape of loaded EEG data object: {eeg_subject_trials.shape}")
        if len(eeg_subject_trials) > 0:
            print(f"Type of first EEG trial: {type(eeg_subject_trials[0])}")
            print(f"Shape of first EEG trial: {eeg_subject_trials[0].shape}")
            print(f"Shape of first Kinematic trial: {kin_subject_trials[0].shape}")
        print("---------------------------------\n")


    except FileNotFoundError as e:
        print(f"Error: Could not find data file -> {e}")
        exit()

    # --- Step 2: Split the Subject's Trials into Train, Validation, and Test Sets ---
    n_trials = len(eeg_subject_trials)
    trial_indices = list(range(n_trials))
    n_train = int(n_trials * TRAIN_RATIO)
    n_val = int(math.ceil((n_trials * VALIDATION_RATIO)))
    train_indices = trial_indices[:n_train]
    test_indices = trial_indices[n_train : n_train + n_val]
    val_indices = trial_indices[n_train + n_val:]

    print(f'VAL INDICES{val_indices}')

    eeg_train_trials = [eeg_subject_trials[i] for i in train_indices]
    kin_train_trials = [kin_subject_trials[i] for i in train_indices]
    eeg_val_trials = [eeg_subject_trials[i] for i in val_indices]
    kin_val_trials = [kin_subject_trials[i] for i in val_indices]
    eeg_test_trials = [eeg_subject_trials[i] for i in test_indices]
    kin_test_trials = [kin_subject_trials[i] for i in test_indices]

    print("\n--- Data Split (by trial) ---")
    print(f"Total trials for subject: {n_trials}")
    print(f"Training trials: {len(eeg_train_trials)}")
    print(f"Validation trials: {len(eeg_val_trials)}")
    print(f"Testing trials: {len(eeg_test_trials)}")


    # --- Step 3: Sliding Window Function (Corrected) ---

    def create_windowed_samples(eeg_trials, kin_trials, window_size, step_size):
        eeg_samples, kin_samples = [], []
        
        for eeg, kin in zip(eeg_trials, kin_trials):
            min_len = min(eeg.shape[1], kin.shape[0])
            
            eeg = eeg[:, :min_len]
            kin = kin[:min_len, :]
            
            if eeg.shape[1] >= window_size:
                for i in range(0, eeg.shape[1] - window_size + 1, step_size):
                    eeg_samples.append(eeg[:, i : i + window_size])
                    kin_samples.append(kin[i : i + window_size, :])
                
        return np.array(eeg_samples), np.array(kin_samples)


    # --- Step 4: Create Windowed Datasets for Each Split ---
    print("\nCreating windowed samples for each data split...")

    X_train, y_train = create_windowed_samples(eeg_train_trials, kin_train_trials, WINDOW_SIZE, STEP_SIZE)
    X_val, y_val = create_windowed_samples(eeg_val_trials, kin_val_trials, WINDOW_SIZE, STEP_SIZE)
    # STEP_SIZE=
    X_test, y_test = create_windowed_samples(eeg_test_trials, kin_test_trials, WINDOW_SIZE, WINDOW_SIZE)
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")


    # --- Step 5: Normalize the Data (Fit on Training Set Only) ---
    print("\nNormalizing data...")

    # --- 5a. Kinematic Data (y): Min-Max Scaling ---
    n_samples_train, n_timesteps_train, n_features_kin = y_train.shape
    y_train_reshaped = y_train.reshape(-1, n_features_kin)

    scaler_kin = MinMaxScaler(feature_range=(0, 1))
    scaler_kin.fit(y_train_reshaped)

    y_train = scaler_kin.transform(y_train_reshaped).reshape(n_samples_train, n_timesteps_train, n_features_kin)
    if y_val.shape[0] > 0:
        n_samples_val, n_timesteps_val, _ = y_val.shape
        y_val = scaler_kin.transform(y_val.reshape(-1, n_features_kin)).reshape(n_samples_val, n_timesteps_val, n_features_kin)
    if y_test.shape[0] > 0:
        n_samples_test, n_timesteps_test, _ = y_test.shape
        y_test = scaler_kin.transform(y_test.reshape(-1, n_features_kin)).reshape(n_samples_test, n_timesteps_test, n_features_kin)
    print("Kinematic data (y) normalized using MinMaxScaler.")  

    # --- 5b. EEG Data (X): Z-Score Normalization (Per Channel) ---
    if X_train.shape[0] > 0:
        # Reshape (samples, channels, timesteps) -> (channels, samples * timesteps)
        # This combines all time points for each channel to calculate stats.
        n_samples_train_eeg, n_channels_eeg, n_timesteps_train_eeg = X_train.shape
        X_train_reshaped = X_train.transpose(1, 0, 2).reshape(n_channels_eeg, -1)

        eeg_mean = X_train_reshaped.mean(axis=1).reshape(-1, 1) # Shape: (n_channels, 1)
        eeg_std = X_train_reshaped.std(axis=1).reshape(-1, 1)   # Shape: (n_channels, 1)
        epsilon = 1e-8 # To prevent division by zero

        # Normalize using the calculated mean and std for all sets
        X_train = (X_train - eeg_mean) / (eeg_std + epsilon)
        if X_val.shape[0] > 0:
            X_val = (X_val - eeg_mean) / (eeg_std + epsilon)
        if X_test.shape[0] > 0:
            X_test = (X_test - eeg_mean) / (eeg_std + epsilon)
        print("EEG data (X) normalized using Z-Score.")
    else:
        print("WARNING: No training data for EEG scaler. Skipping normalization.")


    # --- Step 6: Create PyTorch DataLoaders ---
    # NOTE: The EEG data shape is (samples, channels, time), which is the
    # expected format for layers like Conv1d and BatchNorm1d.
    def create_dataloader(X, y, batch_size, shuffle=False):
        """Converts NumPy arrays to a PyTorch DataLoader."""
        if X.shape[0] == 0:
            return None
        X_tensor = torch.tensor(X).float()
        y_tensor = torch.tensor(y).float()
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader

    print("\nCreating PyTorch DataLoaders...")
    train_loader = create_dataloader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, BATCH_SIZE)
    test_loader = create_dataloader(X_test, y_test, BATCH_SIZE)

    print(f"DataLoaders created with batch size {BATCH_SIZE}.")

    # --- Example of how to use the DataLoader ---
    print("\n--- Example Batch from Training Loader ---")
    if train_loader:
        eeg_batch, kin_batch = next(iter(train_loader))
        print(f"Shape of one EEG batch (X): {eeg_batch.shape}")
        print(f"Shape of one Kinematic batch (y): {kin_batch.shape}")
    else:
        print("Training loader is empty.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    class PreMovNet(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(PreMovNet, self).__init__()

            self.batch_norm = nn.BatchNorm1d(input_dim)
            # Calculate padding for 'same' padding (keeping the output size same as input size)
            self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=7, padding=3)  # equivalent to "same"
            self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, padding=2)  # equivalent to "same"
            self.maxpool1 = nn.MaxPool1d(kernel_size=5, stride=1, padding=2)
            self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
            self.dropout = nn.Dropout(0.25)
            self.lstm = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)

            self.fc1 = nn.Linear(128, 128)
            self.fc2 = nn.Linear(128, output_dim)

        def forward(self, x):
            x = self.batch_norm(x)
            
            x = torch.relu(self.conv1(x))  # Apply first convolution
            x = self.maxpool1(x)  # Apply first maxpooling
            x = self.dropout(x)  # Apply dropout
            x = torch.relu(self.conv2(x))  # Apply second convolution
            x = self.maxpool2(x)  # Apply second maxpooling
            
            x = x.permute(0, 2, 1)  # Swap back (batch, features, timesteps) → (batch, timesteps, features)
            x, _ = self.lstm(x)  # LSTM expects (batch, timesteps, features)

            x = torch.relu(self.fc1(x))  # Apply fully connected layer to all timesteps
            x = self.fc2(x)  # Output kinematics for each timestep

            return x


    input_dim = 60  # Number of features
    output_dim = 3 # Number of target values

    time_steps = 500

    model = PreMovNet(input_dim,output_dim).to(device)



    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-2)
    print(input_dim)
    print(output_dim)

    



    train_losses = []
    val_losses = []

    def train_one_epoch(model, train_loader, val_loader, criterion, optimizer, device, epoch, num_epochs):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0.0
        train_batches = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            target = batch_y  # Shape changes from (64, 1000, 18) to (64, 18)
            loss = criterion(output, target)        
            # loss = criterion(output, batch_y)

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"  WARNING: Invalid loss detected during TRAINING! Epoch {epoch+1}. Skipping batch update.")
                # Optionally skip optimizer step if loss is bad
                continue # Skip to next batch

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_batches += 1

        # Avoid division by zero if train_loader is empty or all batches had NaN loss
        avg_train_loss = total_train_loss / train_batches if train_batches > 0 else None

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
                val_output = model(batch_X_val)
                val_target = batch_y_val# Shape changes from (64, 1000, 18) to (64, 18)
                # val_loss = criterion(val_output, batch_y_val)
                val_loss = criterion(val_output, val_target)            






                if torch.isnan(val_loss).any() or torch.isinf(val_loss).any():
                    print(f"  WARNING: Invalid loss detected during VALIDATION! Epoch {epoch+1}. Skipping batch.")
                    continue # Skip accumulating invalid loss

                total_val_loss += val_loss.item()
                val_batches +=  1

        # Avoid division by zero if val_loader is empty or all batches had NaN loss
        avg_val_loss = total_val_loss / val_batches if val_batches > 0 else None


        return avg_train_loss, avg_val_loss


    # =============================================
    # -------------------------------------------------------------------


    # --- Early Stopping Parameters ---
    patience = 20  # How many epochs to wait after last improvement
    best_val_loss = np.inf # Initialize best validation loss to infinity
    epochs_no_improve = 0  # Counter for epochs without improvement
    best_model_state = None # To store the state_dict of the best model

    # Training loop
    num_epochs = 200 # Keep the original maximum epochs
    # Ensure 'model' is defined before this print statement
    print(f"Starting training for {type(model).__name__} targeting  (max {num_epochs} epochs, patience={patience})...")

    for epoch in range(num_epochs):
        # Call the training function for one epoch, passing  -specific loaders
        avg_train_loss, avg_val_loss = train_one_epoch(
            model, train_loader, val_loader, criterion, optimizer, device, epoch, num_epochs
        )
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Check if training/validation failed for this epoch
        if avg_train_loss is None or avg_val_loss is None:
            print(f"Stopping training due to errors in epoch {epoch+1}.")
            break

        # Print the average losses for this epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # --- Early Stopping Check ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save the best model state
            best_model_state = copy.deepcopy(model.state_dict()) # Make a deep copy
            print(f"  Validation loss improved. Saving model state at epoch {epoch+1}")
        else:
            epochs_no_improve += 1

            print(f"  Validation loss did not improve for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs!")
            print(f"Best validation loss achieved: {best_val_loss:.6f}")
            break # Exit the training loop

    # --- End of Training ---
    print("\nTraining finished.")

    # --- Load the best model state ---
    if best_model_state is not None:
        print("Loading best model weights found during training.")
        model.load_state_dict(best_model_state)
        # Optional: Save the best model state to file
        save_path = f"best_{type(model).__name__}__state.pth" # Example filename for  model
        torch.save(best_model_state, save_path)
        print(f"Best model state saved to {save_path}")
    else:
        print("Warning: No best model state was saved (perhaps training stopped early or validation loss never improved).")

    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    folder_name = f"results_premovnet_0.1-40leftfoothz"
    os.makedirs(folder_name, exist_ok=True)

    file_name = f"participants_{TARGET_SUBJECT_KEY}_shift_{STEP_SIZE}_training_validation_loss_{WINDOW_SIZE}ms_window.png"
    save_path = os.path.join(folder_name, file_name)

    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as {file_name}")
    except Exception as e_save:
        print(f"Error saving plot: {e_save}")

    









    
    # --- 1. Prediction Phase ---
    print("--- Starting Evaluation on Test Set ---")
    # model.eval()
    all_predictions = []
    all_targets = []

    if test_loader:
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                predictions = model(batch_X)
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        print("Prediction phase complete.")



    final_predictions = np.concatenate(all_predictions, axis=0)
    final_targets = np.concatenate(all_targets, axis=0)

    print(f"\nShape of final windowed predictions array: {final_predictions.shape}")
    print(f"Shape of final windowed targets array: {final_targets.shape}")


    # --- 2. Performance Metrics Calculation (PCC, RMSE, MAE) ---
    print("\n--- Calculating Performance Metrics ---")
    num_windows, window_len, num_features = final_predictions.shape
    predictions_flat = final_predictions.reshape(-1, num_features)
    targets_flat = final_targets.reshape(-1, num_features)

    pcc_results, mse_results, mae_results = {}, {}, {}
    pcc_text_lines, mse_text_lines, mae_text_lines = [], [], []
    dimensions = ['X', 'Y', 'Z']

    for i in range(num_features):
        dim_name = dimensions[i] if i < len(dimensions) else f"Feature_{i+1}"
        pred_dim = predictions_flat[:, i]
        target_dim = targets_flat[:, i]

        rmse = sqrt(mean_squared_error(target_dim, pred_dim))
        mae = mean_absolute_error(target_dim, pred_dim)
        corr, p_value = pearsonr(pred_dim, target_dim)
        
        mse_results[dim_name] = rmse
        mae_results[dim_name] = mae
        pcc_results[dim_name] = (corr, p_value)
        
        mse_text_lines.append(f"RMSE ({dim_name}): {rmse:.3f}")
        pcc_text_lines.append(f"PCC ({dim_name}): {corr:.3f}")
        mae_text_lines.append(f"MAE ({dim_name}): {mae:.3f}")

    avg_mae = np.mean(list(mae_results.values()))
    avg_rmse = np.mean(list(mse_results.values()))
    valid_corrs = [r[0] for r in pcc_results.values() if not np.isnan(r[0])]
    avg_pcc = np.mean(valid_corrs) if valid_corrs else np.nan

    mae_text_lines.append(f"Average MAE: {avg_mae:.3f}")
    mse_text_lines.append(f"Average RMSE: {avg_rmse:.3f}")
    pcc_text_lines.append(f"Average PCC: {avg_pcc:.3f}")


    
    with open(f'CNNLSTM_0.1-40hz_RightUpperLeg.csv', 'a') as fd:
        correlations = [r[0] for r in (list(pcc_results.values()))]
        distances=list(mse_results.values())
        distances_a=list(mae_results.values())

        line_to_write = f"participant:{TARGET_SUBJECT_KEY}\nPCC:"+",".join([f"{corr:.3f}" for corr in correlations]) +f",{avg_pcc:.3f}"+"\n"
        line_to_write += f"RMSE:"+",".join([f"{dist:.3f}" for dist in distances]) +f",{avg_rmse:.3f}"+"\n"
        line_to_write += f"MAE:"+",".join([f"{dist:.3f}" for dist in distances_a]) +f",{avg_mae:.3f}"+"\n"

        fd.write(line_to_write)

    # --- 3. Plotting Continuous Results (Corrected Method) ---
    print("\n--- Plotting Continuous Results ---")

    # Check if there is anything in the test loader to plot
    if 'test_loader' in locals() and test_loader is not None and len(all_targets) > 0:
        
        # Reshape the non-overlapping windows into a single continuous signal.
        # This is the correct way to reconstruct the time series from your test windows.
        # Shape changes from (num_windows, window_len, features) to (total_timesteps, features)
        continuous_predictions = final_predictions.reshape(-1, num_features)
        continuous_targets = final_targets.reshape(-1, num_features)

        # --- Plotting Configuration ---
        duration_to_plot_sec = 20
        sampling_rate_hz = 1000 # As defined in your script
        
        # 1. Create a single time axis for the *entire* continuous signal
        total_timesteps = continuous_targets.shape[0]
        full_time_axis = np.arange(total_timesteps) / sampling_rate_hz
        
        # 2. Determine how many data points to plot (up to the desired duration)
        timesteps_to_plot = min(total_timesteps, int(duration_to_plot_sec * sampling_rate_hz))
        
        # 3. Slice the data and the time axis to the same length
        plot_targets = continuous_targets[:timesteps_to_plot]
        plot_preds = continuous_predictions[:timesteps_to_plot]
        plot_time_axis = full_time_axis[:timesteps_to_plot]
        
        # --- Create the Plot ---
        fig, axes = plt.subplots(num_features, 1, figsize=(15, num_features * 3), sharex=True, squeeze=False)
        fig.suptitle(f"Continuous Kinematics (First {plot_time_axis[-1]:.2f}s): Predicted vs. Ground Truth", fontsize=16)
        
        # This loop now iterates only through the features (X, Y, Z), not the windows
        for i in range(num_features):
            ax = axes[i, 0]
            dim_label = dimensions[i] if i < len(dimensions) else f"Feature_{i+1}"
            
            # Plot the continuous data directly
            ax.plot(plot_time_axis, plot_targets[:, i], color='black', label='Ground Truth', linewidth=1.5)
            ax.plot(plot_time_axis, plot_preds[:, i], color='red', linestyle='--', label='Predicted', linewidth=1.0)

            ax.set_title(f"{dim_label} Trajectory")
            ax.set_ylabel("Value")
            ax.grid(True, linestyle=':')
            ax.set_ylim(0, 1)

        # Add a single legend to the first plot
        axes[0, 0].legend()
        axes[-1, 0].set_xlabel("Time (seconds)")
        
        # Add metrics text box
        mae_full_text = "\n".join(mae_text_lines)
        pcc_full_text = "\n".join(pcc_text_lines)
        mse_full_text = "\n".join(mse_text_lines)
        bbox_props = dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7)
        fig.text(0.37, 0.01, mae_full_text, ha='center', va='bottom', fontsize=9, bbox=bbox_props)
        fig.text(0.47, 0.01, pcc_full_text, ha='center', va='bottom', fontsize=9, bbox=bbox_props)
        fig.text(0.57, 0.01, mse_full_text, ha='center', va='bottom', fontsize=9, bbox=bbox_props)
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.96])

        file_name = f"participant_{TARGET_SUBJECT_KEY}_cnnlstm.png"
        save_path = os.path.join(folder_name, file_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as {file_name}")

    else:
        print("No test data was generated, skipping plot.")
