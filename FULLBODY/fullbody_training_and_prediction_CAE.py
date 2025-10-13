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

print("All modules imported successfully!")

#-----------------------------------------------------------------------------
# --- 1. DEFINE THE NEW MODEL BASED ON SCG_CAE ARCHITECTURE ---
# This model replaces the previous 1D U-Net.
# It has been generalized to accept multi-channel input and output.
#-----------------------------------------------------------------------------

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class CAE(nn.Module):
    def __init__(self, input_channels=60, output_channels=3):
        super(CAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=128, kernel_size=8, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=8, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=8, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=8, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(in_channels=1024, out_channels=256, kernel_size=8, stride=1, padding='same'),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=8, stride=1, padding='same'),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=8, stride=1, padding='same'),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2),
            nn.Conv1d(in_channels=64, out_channels=output_channels, kernel_size=8, stride=1, padding='same'),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent_representation = self.encoder(x)
        reconstructed_signal = self.decoder(latent_representation)
        return reconstructed_signal



#-----------------------------------------------------------------------------
# --- 2. SETUP GLOBAL PARAMETERS & HYPERPARAMETERS ---
#-----------------------------------------------------------------------------
DATA_FOLDER="RightUpperLeg" 
WINDOW_SIZE = 512      # Input sequence length
STEP_SIZE = int(0.5*(WINDOW_SIZE))  # 50% overlap for training/validation windows

# Model Hyperparameters
INPUT_CHANNELS = 60 # Number of EEG channels
OUTPUT_CHANNELS = 3 # Number of kinematic features
LEARNING_RATE = 0.0001
N_EPOCHS = 50
BATCH_SIZE = 64

# Train/Validation/Test Split Ratios
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
# TEST_RATIO is implicitly 1.0 - TRAIN_RATIO - VALIDATION_RATIO

#-----------------------------------------------------------------------------
# --- MAIN SCRIPT ---
#-----------------------------------------------------------------------------

for TARGET_SUBJECT_KEY in range(1, 10):
    print(f"\n{'='*80}\nStarting processing for subject key: {TARGET_SUBJECT_KEY}\n{'='*80}")
    
    # --- Step 1: Data Loading (remains the same) ---
    print(f"Loading pre-processed data for subject key: {TARGET_SUBJECT_KEY}...")
    try:
        with open(f"RightUpperLeg/0.1-40hz_filtered_eeg_data_for_subject{TARGET_SUBJECT_KEY}", 'rb') as f:
            eeg_subject_trials = pickle.load(f)

        kin_file_path = os.path.join(DATA_FOLDER, f"kin_data_for_subject{TARGET_SUBJECT_KEY}.npy")
        kin_subject_trials = np.load(kin_file_path, allow_pickle=True)
        
        print(f"Successfully loaded {len(eeg_subject_trials)} trials for the target subject.")
    except FileNotFoundError as e:
        print(f"Error: Could not find data file -> {e}")
        continue # Skip to the next subject

    # --- Step 2: Data Splitting (remains the same) ---
    n_trials = len(eeg_subject_trials)
    trial_indices = list(range(n_trials))
    n_train = int(n_trials * TRAIN_RATIO)
    n_val = int(math.ceil((n_trials * VALIDATION_RATIO)))
    train_indices, test_indices, val_indices = trial_indices[:n_train], trial_indices[n_train:n_train + n_val], trial_indices[n_train + n_val:]

    eeg_train_trials, kin_train_trials = [eeg_subject_trials[i] for i in train_indices], [kin_subject_trials[i] for i in train_indices]
    eeg_val_trials, kin_val_trials = [eeg_subject_trials[i] for i in val_indices], [kin_subject_trials[i] for i in val_indices]
    eeg_test_trials, kin_test_trials = [eeg_subject_trials[i] for i in test_indices], [kin_subject_trials[i] for i in test_indices]

    print("\n--- Data Split (by trial) ---")
    print(f"Training: {len(eeg_train_trials)}, Validation: {len(eeg_val_trials)}, Testing: {len(eeg_test_trials)}")

    # --- Step 3: Sliding Window Function (remains the same) ---
    def create_windowed_samples(eeg_trials, kin_trials, window_size, step_size):
        eeg_samples, kin_samples = [], []
        for eeg, kin in zip(eeg_trials, kin_trials):
            min_len = min(eeg.shape[1], kin.shape[0])
            eeg, kin = eeg[:, :min_len], kin[:min_len, :]
            if eeg.shape[1] >= window_size:
                for i in range(0, eeg.shape[1] - window_size + 1, step_size):
                    eeg_samples.append(eeg[:, i : i + window_size])
                    kin_samples.append(kin[i : i + window_size, :])
        return np.array(eeg_samples), np.array(kin_samples)

    # --- Step 4: Create Windowed Datasets ---
    print("\nCreating windowed samples...")
    X_train, y_train = create_windowed_samples(eeg_train_trials, kin_train_trials, WINDOW_SIZE, STEP_SIZE)
    X_val, y_val = create_windowed_samples(eeg_val_trials, kin_val_trials, WINDOW_SIZE, STEP_SIZE)
    # --- Using NON-OVERLAPPING windows for the test set ---
    X_test, y_test = create_windowed_samples(eeg_test_trials, kin_test_trials, WINDOW_SIZE, WINDOW_SIZE)
    
    # Check for empty sets
    if X_train.shape[0] == 0 or X_val.shape[0] == 0:
        print("Not enough data to create train/validation splits. Skipping subject.")
        continue

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Testing samples (non-overlapping): {X_test.shape[0]}")

    # --- Step 5: Normalization ---
    print("\nNormalizing data...")
    # Kinematic Data (y): Min-Max Scaling to [-1, 1] to match model's Tanh output
    n_samples_train, n_timesteps_train, n_features_kin = y_train.shape
    y_train_reshaped = y_train.reshape(-1, n_features_kin)
    scaler_kin = MinMaxScaler(feature_range=(0, 1)) # Range is now -1 to 1
    scaler_kin.fit(y_train_reshaped)
    y_train = scaler_kin.transform(y_train_reshaped).reshape(n_samples_train, n_timesteps_train, n_features_kin)
    n_samples_val, n_timesteps_val, _ = y_val.shape
    y_val = scaler_kin.transform(y_val.reshape(-1, n_features_kin)).reshape(n_samples_val, n_timesteps_val, n_features_kin)
    if y_test.shape[0] > 0:
        n_samples_test, n_timesteps_test, _ = y_test.shape
        y_test = scaler_kin.transform(y_test.reshape(-1, n_features_kin)).reshape(n_samples_test, n_timesteps_test, n_features_kin)
    
    # EEG Data (X): Z-Score Normalization
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






    # --- Step 6: Create PyTorch DataLoaders (remains the same) ---
    def create_dataloader(X, y, batch_size, shuffle=False):
        if X.shape[0] == 0: return None
        return DataLoader(TensorDataset(torch.tensor(X).float(), torch.tensor(y).float()), batch_size=batch_size, shuffle=shuffle)
    
    train_loader = create_dataloader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, BATCH_SIZE)
    test_loader = create_dataloader(X_test, y_test, BATCH_SIZE)
    print("PyTorch DataLoaders created.")

    # --- 7. Initialize Model, Loss, and Optimizer ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # <<< MODEL INSTANTIATION IS UPDATED HERE >>>
    model = CAE(input_channels=INPUT_CHANNELS, output_channels=OUTPUT_CHANNELS).to(device)
    
    criterion = nn.L1Loss() # Mean Absolute Error
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Model [{type(model).__name__}] was created successfully.")

    # --- 8. Training and Validation Loop ---
    loss_history = {"train": [], "val": []}
    patience=20
    best_val_loss = np.inf # Initialize best validation loss to infinity
    epochs_no_improve = 0  # Counter for epochs without improvement
    best_model_state = None # To store the state_dict of the best model


    for epoch in range(N_EPOCHS):
        epoch_start_time = time.time()
        
        # --- TRAINING ---
        model.train()
        running_train_loss = 0.0
        for eeg_batch, kin_batch in train_loader:
            eeg_batch, kin_batch = eeg_batch.to(device), kin_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(eeg_batch)
            targets = kin_batch.permute(0, 2, 1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            
        avg_train_loss = running_train_loss / len(train_loader)
        loss_history["train"].append(avg_train_loss)

        # --- VALIDATION ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for eeg_batch, kin_batch in val_loader:
                eeg_batch, kin_batch = eeg_batch.to(device), kin_batch.to(device)
                outputs = model(eeg_batch)
                targets = kin_batch.permute(0, 2, 1)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item()
                
        avg_val_loss = running_val_loss / len(val_loader)
        loss_history["val"].append(avg_val_loss)
        




        print(f"Epoch [{epoch+1}/{N_EPOCHS}] | Time: {time.time() - epoch_start_time:.2f}s | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

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
        # torch.save(best_model_state, save_path)
        # print(f"Best model state saved to {save_path}")
    else:
        print("Warning: No best model state was saved (perhaps training stopped early or validation loss never improved).")















    # --- 9. Plotting for Training/Validation Loss ---
    print("\nTraining complete. Plotting losses...")
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history['train'], label='Training L1 Loss')
    plt.plot(loss_history['val'], label='Validation L1 Loss')
    plt.title(f'Loss History for Subject {TARGET_SUBJECT_KEY}')
    plt.xlabel('Epochs')
    plt.ylabel('L1 (MAE) Loss')
    plt.legend()
    plt.grid(True, linestyle=':')
    
    folder_name = "Convolutional_Autoencoder_0.1-40hz" # Changed folder name
    os.makedirs(folder_name, exist_ok=True)
    file_name = f"participant_{TARGET_SUBJECT_KEY}_loss_plot.png"
    plt.savefig(os.path.join(folder_name, file_name))
    plt.close()
    print(f"Loss plot saved to {os.path.join(folder_name, file_name)}")
    
    # --- 10. Final Evaluation on Test Set ---
    print("\n--- Starting Evaluation on Test Set ---")
    model.eval()
    all_predictions, all_targets = [], []
    
    if test_loader:
        with torch.no_grad():
            for eeg_batch, kin_batch in test_loader:
                eeg_batch = eeg_batch.to(device)
                outputs = model(eeg_batch)
                
                print(f"Testing samples (outputs): {outputs.shape}\n")
                print(f"Testing samples (kin_Batch): {kin_batch.shape}")

                all_predictions.append(outputs.permute(0, 2, 1).cpu().numpy())
                all_targets.append(kin_batch.cpu().numpy())
    
        final_predictions = np.concatenate(all_predictions, axis=0)
        final_targets = np.concatenate(all_targets, axis=0)
                
        print(f"Shape of final predictions array: {final_predictions.shape}")

        # --- Performance Metrics Calculation ---
        print("\n--- Calculating Performance Metrics ---")
        num_windows, window_len, num_features = final_predictions.shape
        predictions_flat = final_predictions.reshape(-1, num_features)
        targets_flat = final_targets.reshape(-1, num_features)

        pcc_results, mse_results, mae_results = {}, {}, {}
        pcc_text_lines, mse_text_lines, mae_text_lines = [], [], []
        dimensions = ['X', 'Y', 'Z']

        for i,dim_name in enumerate(dimensions):
            pred_dim, target_dim = predictions_flat[:, i], targets_flat[:, i]
            rmse = sqrt(mean_squared_error(target_dim, pred_dim))
            mae = mean_absolute_error(target_dim, pred_dim)
            corr, p_value = pearsonr(pred_dim, target_dim)
            mse_results[dim_name], mae_results[dim_name], pcc_results[dim_name] = rmse, mae, (corr, p_value)
            mse_text_lines.append(f"RMSE ({dim_name}): {rmse:.3f}")
            pcc_text_lines.append(f"PCC ({dim_name}): {corr:.3f}")
            mae_text_lines.append(f"MAE ({dim_name}): {mae:.3f}")

        avg_mae, avg_rmse = np.mean(list(mae_results.values())), np.mean(list(mse_results.values()))
        valid_corrs = [r[0] for r in pcc_results.values() if not np.isnan(r[0])]
        avg_pcc = np.mean(valid_corrs) if valid_corrs else np.nan
        mae_text_lines.append(f"\nAverage MAE: {avg_mae:.3f}")
        mse_text_lines.append(f"\nAverage RMSE: {avg_rmse:.3f}")
        pcc_text_lines.append(f"\nAverage PCC: {avg_pcc:.3f}")



            
        with open(f'CAE_0.1-40hz.csv', 'a') as fd:
            correlations = [r[0] for r in (list(pcc_results.values()))]
            distances=list(mse_results.values())
            distances_a=list(mae_results.values())

            line_to_write = f"participant:{TARGET_SUBJECT_KEY}\nPCC:"+",".join([f"{corr:.3f}" for corr in correlations]) +f",{avg_pcc:.3f}"+"\n"
            line_to_write += f"RMSE:"+",".join([f"{dist:.3f}" for dist in distances]) +f",{avg_rmse:.3f}"+"\n"
            line_to_write += f"MAE:"+",".join([f"{dist:.3f}" for dist in distances_a]) +f",{avg_mae:.3f}"+"\n"

            fd.write(line_to_write)


        max_time_to_plot_sec=20
        sampling_rate=1000
        # --- Plotting Final Non-Overlapping Predictions ---
        print("\n--- Plotting Non-Overlapping Results ---")
        fig, axes = plt.subplots(num_features, 1, figsize=(15, num_features * 3), sharex=True, squeeze=False)
        total_plot_duration_samples = max_time_to_plot_sec*sampling_rate
        time_axis = np.linspace(0, total_plot_duration_samples / 1000, total_plot_duration_samples)
        
        fig.suptitle(f"Continuous Kinematics for Subject {TARGET_SUBJECT_KEY}: Predicted vs. Ground Truth", fontsize=16)
        
        for i,dim_name in enumerate(dimensions):
            ax = axes[i, 0]
            ax.plot(time_axis, targets_flat[:total_plot_duration_samples, i], color='black', label='Ground Truth', linewidth=1.5)
            ax.plot(time_axis, predictions_flat[:total_plot_duration_samples, i], color='red', linestyle='--', label='Predicted',linewidth=1.0)
            ax.set_title(f"{dim_name} Axis")
            ax.set_ylabel("Value")
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle=':')

        axes[0, 0].legend()
        axes[-1, 0].set_xlabel("Time (seconds)")
        
        mae_full_text = "\n".join(mae_text_lines)
        pcc_full_text = "\n".join(pcc_text_lines)
        mse_full_text = "\n".join(mse_text_lines)
        bbox_props = dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7)
        fig.text(0.37, 0.01, mae_full_text, ha='center', va='bottom', fontsize=9, bbox=bbox_props)
        fig.text(0.47, 0.01, pcc_full_text, ha='center', va='bottom', fontsize=9, bbox=bbox_props)
        fig.text(0.57, 0.01, mse_full_text, ha='center', va='bottom', fontsize=9, bbox=bbox_props)
        plt.tight_layout(rect=[0, 0.08, 1, 0.96])

        file_name = f"participant_{TARGET_SUBJECT_KEY}_prediction_plot.png"
        plt.savefig(os.path.join(folder_name, file_name))
        print(f"Prediction plot saved to {os.path.join(folder_name, file_name)}")
        plt.close()

    else:
        print("Test loader is empty. No evaluation was performed.")

print(f"\n{'='*80}\nAll subjects processed.\n{'='*80}")