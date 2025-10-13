
  
import os
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import resample
import scipy.io
import numpy as np
import gc
import mne
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, TensorDataset, DataLoader
import torch.optim as optim
import copy
from scipy.stats import pearsonr
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys # For command-line arguments
import time
import random
from torch.utils.data import TensorDataset, DataLoader
print("All modules installed and imported successfully!")
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error



try:    
    sys.argv[1]
except NameError:
    participant_arg= 3
else:
    participant_arg= int(sys.argv[1])

TARGET_SUBJECT_KEY=participant_arg



kin_data = np.load("kin_data_position.npy", allow_pickle=True)
wrist_kin=[]
wrist_indices = [2, 5, 8]

for session in kin_data:
    wrist_kin.append(session[:,wrist_indices])
kin_data=wrist_kin

# --- Configuration ---
# participant_arg = 3
data_root_dir = "."
eeg_fif_path = "filtered_noica_1-40.fif"


sfreq = 500  # Hz

# --- Windowing Parameters ---
window_length_sec = 0.512
window_length_samples = int(window_length_sec * sfreq) # 256
train_val_step_samples = int(round(0.5*(window_length_samples)))  # Overlapping stride
test_step_samples = window_length_samples # Non-overlapping stride

# --- Define Split Percentages for Runs ---
train_percentage = 0.80
val_percentage = 0.10
lag_s=0

# --- Load Kinematic Data ---

# --- Load EEG Data ---
print(f"Loading EEG data from {eeg_fif_path}...")
try:
    eeg_raw_alldata = mne.io.read_raw_fif(eeg_fif_path, preload=False, verbose='WARNING')
    final_channels = ['F3', 'Fz', 'F4', 'FC5', 'FC1', 'FC2', 'FC6',
                  'C3', 'Cz', 'C4', 'CP5', 'CP1', 'CP2', 'CP6',
                  'P7', 'P3', 'Pz', 'P4', 'O1', 'Oz', 'O2']
    eeg_raw_alldata.pick_channels(final_channels)

except Exception as e:
    print(f"ERROR: Could not load EEG data. {e}")
    exit()

kin_session_lengths_timesteps = [session.shape[0] for session in kin_data]
eeg_session_start_indices = [0] + list(np.cumsum(kin_session_lengths_timesteps[:-1]))

# --- Initialize lists for each set ---
X_train, y_train = [], []
X_val, y_val = [], []
X_test, y_test = [], []

# --- Run-based Splitting Logic ---

# Identify all runs for the target participant
participant_run_indices = [i for i, _ in enumerate(kin_data) if (i // 9) + 1 == participant_arg]
print(f"Found {len(participant_run_indices)} runs for Participant {participant_arg}.")
num_runs = len(participant_run_indices)

train_end_idx = int(num_runs * train_percentage)
val_end_idx = int(num_runs * (train_percentage + val_percentage)) # This is the key change

train_run_indices = participant_run_indices[:train_end_idx]
val_run_indices = participant_run_indices[train_end_idx:val_end_idx]
test_run_indices = participant_run_indices[val_end_idx:]


print(f"Splitting runs: {len(train_run_indices)} train, {len(val_run_indices)} val, {len(test_run_indices)} test")

# --- Efficiently load the .mat file ONCE for the participant ---
print(f"Loading .mat file for Participant {participant_arg}...")
participant_folder = os.path.join(data_root_dir, f"P{participant_arg}")
all_lifts_path = os.path.join(participant_folder, f"P{participant_arg}_AllLifts.mat")
mat_data = scipy.io.loadmat(all_lifts_path)
p_lifts_matrix = mat_data['P']['AllLifts'][0, 0]
del mat_data # Free up memory
gc.collect()

# --- Helper function to process all trials within a given run ---
def process_run(global_session_idx, step_size, X_list, y_list):
    participant_id = (global_session_idx // 9) + 1
    run_number = (global_session_idx % 9) + 1
    
    current_run_kin_data = kin_data[global_session_idx]
    current_session_global_eeg_start_sample = eeg_session_start_indices[global_session_idx]
    all_trials_in_run = p_lifts_matrix[(p_lifts_matrix[:, 0] == participant_id) & (p_lifts_matrix[:, 1] == run_number)]

    for trial_row in all_trials_in_run:
        start_time_in_run_sec = trial_row[7]
        thandstart_relative_sec = trial_row[33]
        thandstop_relative_sec = trial_row[34]
        kin_lift_start_sample = int(round((start_time_in_run_sec + thandstart_relative_sec) * sfreq))
        kin_lift_end_sample = int(round((start_time_in_run_sec + thandstop_relative_sec) * sfreq))

        for window_start in range(kin_lift_start_sample, kin_lift_end_sample, step_size):
            window_end = window_start + window_length_samples
            if window_end > kin_lift_end_sample:
                break
            if window_end > len(current_run_kin_data):
                continue
            
            kin_window = current_run_kin_data[window_start:window_end, :]
            
            global_eeg_start = current_session_global_eeg_start_sample + window_start-int(round((lag_s*500)))
            global_eeg_end = global_eeg_start + window_length_samples

            if global_eeg_start >= 0 and global_eeg_end <= eeg_raw_alldata.n_times:
                eeg_segment, _ = eeg_raw_alldata[:, global_eeg_start:global_eeg_end]
                
                if kin_window.shape[0] == window_length_samples:# and eeg_segment.shape[1] == window_length_samples:
                    X_list.append(eeg_segment.copy())
                    y_list.append(kin_window.copy())
print("\nGenerating windows for each data set...")

# Process Training Runs (Overlapping)
for session_idx in train_run_indices:
    process_run(session_idx, train_val_step_samples, X_train, y_train)

# Process Validation Runs (Overlapping)
for session_idx in val_run_indices:
    process_run(session_idx, train_val_step_samples, X_val, y_val)

# Process Test Runs (Non-Overlapping)
for session_idx in test_run_indices:
    process_run(session_idx, test_step_samples, X_test, y_test)

# --- Final Counts ---
print("\nData generation and splitting complete.")
print(f"  Total Training Windows generated: {len(X_train)}")
print(f"  Total Validation Windows generated: {len(X_val)}")
print(f"  Total Test Windows generated: {len(X_test)}")


# ---  Convert Data Lists to Single NumPy Arrays ---
print("Converting data lists to NumPy arrays...")
try:
    X_train_np = np.array(X_train, dtype=np.float32)
    y_train_np = np.array(y_train, dtype=np.float32)
    X_val_np   = np.array(X_val, dtype=np.float32)
    y_val_np   = np.array(y_val, dtype=np.float32)
    X_test_np  = np.array(X_test, dtype=np.float32)
    y_test_np  = np.array(y_test, dtype=np.float32)

    print(f"Shape of X_train_np: {X_train_np.shape}")
    print(f"Shape of y_train_np: {y_train_np.shape}")
except Exception as e:
    print(f"ERROR: Could not convert lists to NumPy arrays. Ensure all windows have the same shape. Details: {e}")
    exit()

# --- Kinematic Data (y): MinMax Scaling ---
print("\nNormalizing Kinematic data (MinMaxScaler)...")
if y_train_np.size > 0:
    # Reshape 3D data (samples, timesteps, features) to 2D for scaler
    num_samples, timesteps, features = y_train_np.shape
    y_train_reshaped = y_train_np.reshape(-1, features)

    scaler_kin = MinMaxScaler(feature_range=(0, 1))
    scaler_kin.fit(y_train_reshaped)

    # Transform all datasets and reshape back to 3D
    y_train_scaled_flat = scaler_kin.transform(y_train_np.reshape(-1, features))
    y_train_final = y_train_scaled_flat.reshape(y_train_np.shape)

    y_val_scaled_flat = scaler_kin.transform(y_val_np.reshape(-1, features))
    y_val_final = y_val_scaled_flat.reshape(y_val_np.shape)

    y_test_scaled_flat = scaler_kin.transform(y_test_np.reshape(-1, features))
    y_test_final = y_test_scaled_flat.reshape(y_test_np.shape)

    print("Kinematic data normalized.")
else:
    print("WARNING: Kinematic training data is empty. Skipping kinematic normalization.")
    y_train_final, y_val_final, y_test_final = y_train_np, y_val_np, y_test_np


# --- EEG Data (X): Z-Score Normalization (Per Channel) ---
print("\nNormalizing EEG data (Z-Score per channel)...")
if X_train_np.size > 0:
    # Calculate mean and std across sample and time dimensions for each channel
    # Input shape: (samples, channels, timesteps)
    # Mean/Std shape will be (1, channels, 1) for broadcasting
    eeg_mean = np.mean(X_train_np, axis=(0, 2), keepdims=True)
    eeg_std = np.std(X_train_np, axis=(0, 2), keepdims=True)
    epsilon = 1e-8

    # Normalize using broadcasting (highly efficient)
    X_train_final = (X_train_np - eeg_mean) / (eeg_std + epsilon)
    X_val_final = (X_val_np - eeg_mean) / (eeg_std + epsilon)
    X_test_final = (X_test_np - eeg_mean) / (eeg_std + epsilon)

    print("EEG data normalized.")
else:
    print("WARNING: EEG training data is empty. Skipping EEG normalization.")
    X_train_final, X_val_final, X_test_final = X_train_np, X_val_np, X_test_np

print("\nNormalization process finished.")


# --- 4. Convert to PyTorch Tensors and Create DataLoaders ---
print("\nConverting data to PyTorch Tensors and creating DataLoaders...")

# Data is already in the correct NumPy array format
X_train_tensor = torch.from_numpy(X_train_final)
y_train_tensor = torch.from_numpy(y_train_final)

X_val_tensor = torch.from_numpy(X_val_final)
y_val_tensor = torch.from_numpy(y_val_final)

X_test_tensor = torch.from_numpy(X_test_final)
y_test_tensor = torch.from_numpy(y_test_final)

print(f"Final shape of X_train_tensor: {X_train_tensor.shape}")
print(f"Final shape of y_train_tensor: {y_train_tensor.shape}")

# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset   = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset  = TensorDataset(X_test_tensor, y_test_tensor)

print(f"\nTrain dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

batch_size = 512 # You can adjust this
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

print(f"\nDataLoaders created with batch_size={batch_size}.")
print("Setup of Tensors and DataLoaders is complete.")

# Example: Iterate through a few batches of the training loader
if len(train_loader) > 0:
    print("\nExample of iterating through the train_loader:")
    for i, (eeg_batch, kin_batch) in enumerate(train_loader):
        if i >= 2: # Show first 2 batches
            break
        print(f"\nBatch {i+1}:")
        print(f"  EEG batch shape: {eeg_batch.shape}") 
        print(f"  Kinematics batch shape: {kin_batch.shape}")
else:
    print("Train loader is empty.")



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


# --- 3. Ensure DataLoaders are defined ---
print(f'The number of training samples = {len(train_loader.dataset)}')
print(f'The number of validation samples = {len(val_loader.dataset)}')

INPUT_CHANNELS=21
OUTPUT_CHANNELS=3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE=3e-4
N_EPOCHS=300
# ---  Initialize Model ---
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

folder_name = f"cae_{lag_s*1000}ms_0.5-3hz"
os.makedirs(folder_name, exist_ok=True)
file_name = f"participant_{TARGET_SUBJECT_KEY}_loss_plot.png"
plt.savefig(os.path.join(folder_name, file_name))
plt.close()
print(f"Loss plot saved to {os.path.join(folder_name, file_name)}")






print("\n--- Starting Evaluation on Test Set ---")
model.eval()
all_predictions, all_targets = [], []

if test_loader:
    with torch.no_grad():
        for eeg_batch, kin_batch in test_loader:
            eeg_batch = eeg_batch.to(device)
            # Run the model to get predictions
            outputs = model(eeg_batch)
            print(outputs.shape)
            # Append batches to lists
            # Predictions are (N, C, L), permute to (N, L, C)
            all_predictions.append(outputs.permute(0, 2, 1).cpu().numpy())
            # Targets are already (N, L, C)
            print(kin_batch.shape)
            all_targets.append(kin_batch.cpu().numpy())

    # This concatenates all the trial windows back-to-back
    final_predictions = np.concatenate(all_predictions, axis=0)
    final_targets = np.concatenate(all_targets, axis=0)
            
    print(f"Shape of final windowed predictions array: {final_predictions.shape}")

    # Reshape the stitched windows into one long signal, like in the GAN script
    num_windows, window_len, num_features = final_predictions.shape
    predictions_flat = final_predictions.reshape(-1, num_features)
    targets_flat = final_targets.reshape(-1, num_features)

    # --- Performance Metrics Calculation ---
    print("\n--- Calculating Performance Metrics ---")
    pcc_results, mse_results, mae_results = {}, {}, {}
    pcc_text_lines, mse_text_lines, mae_text_lines = [], [], []
    dimensions = ['X', 'Y', 'Z']

    for i, dim_name in enumerate(dimensions):
        pred_dim, target_dim = predictions_flat[:, i], targets_flat[:, i]
        rmse = sqrt(mean_squared_error(target_dim, pred_dim))
        mae = mean_absolute_error(target_dim, pred_dim)
        corr, _ = pearsonr(pred_dim, target_dim)
        mse_results[dim_name], mae_results[dim_name], pcc_results[dim_name] = rmse, mae, corr
        mse_text_lines.append(f"RMSE ({dim_name}): {rmse:.3f}")
        pcc_text_lines.append(f"PCC ({dim_name}): {corr:.3f}")
        mae_text_lines.append(f"MAE ({dim_name}): {mae:.3f}")

    avg_mae = np.mean(list(mae_results.values()))
    avg_rmse = np.mean(list(mse_results.values()))
    valid_corrs = [r for r in pcc_results.values() if not np.isnan(r)]
    avg_pcc = np.mean(valid_corrs) if valid_corrs else np.nan
    mae_text_lines.append(f"\nAverage MAE: {avg_mae:.3f}")
    mse_text_lines.append(f"\nAverage RMSE: {avg_rmse:.3f}")
    pcc_text_lines.append(f"\nAverage PCC: {avg_pcc:.3f}")


    with open(f'metrics.csv_cae_lag{lag_s*1000}ms_0.5-3hz.csv', 'a') as fd:
        correlations = list(pcc_results.values())
        distances=list(mse_results.values())
        distances_a=list(mae_results.values())

        line_to_write = f"participant:{participant_arg}\nPCC:"+",".join([f"{corr:.3f}" for corr in correlations]) +f",{avg_pcc:.3f}"+"\n"
        line_to_write += f"RMSE:"+",".join([f"{dist:.3f}" for dist in distances]) +f",{avg_rmse:.3f}"+"\n"
        line_to_write += f"MAE:"+",".join([f"{dist:.3f}" for dist in distances_a]) +f",{avg_mae:.3f}"+"\n"

        fd.write(line_to_write)

    # --- Plotting using the robust logic from the GAN script ---
    print("\n--- Plotting Stitched-Trial Results ---")
    fig, axes = plt.subplots(num_features, 1, figsize=(15, num_features * 3), sharex=True, squeeze=False)
    
    # Configuration
    duration_to_plot_sec = 20
    sampling_rate_hz = 500
    
    # 1. Create a time axis for the *entire* stitched signal
    total_timesteps = predictions_flat.shape[0]
    full_time_axis = np.arange(total_timesteps) / sampling_rate_hz
    
    # 2. Determine how many timesteps to actually plot
    timesteps_to_plot = min(total_timesteps, int(duration_to_plot_sec * sampling_rate_hz))
    
    # 3. Slice the data AND the time axis to the same length
    plot_targets = targets_flat[:timesteps_to_plot]
    plot_preds = predictions_flat[:timesteps_to_plot]
    plot_time_axis = full_time_axis[:timesteps_to_plot]
    
    fig.suptitle(f"Continuous Kinematics for Subject {TARGET_SUBJECT_KEY}: Predicted vs. Ground Truth", fontsize=16)
    
    for i, dim_name in enumerate(dimensions):
        ax = axes[i, 0]
        ax.plot(plot_time_axis, plot_targets[:, i], color='black', label='Ground Truth', linewidth=1.5)
        ax.plot(plot_time_axis, plot_preds[:, i], color='red', linestyle='--', label='Predicted', linewidth=1.0)
        ax.set_title(f"{dim_name} Axis")
        ax.set_ylabel("Value")
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle=':')

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

    # Save the figure
    os.makedirs(folder_name, exist_ok=True)
    file_name = f"participant_{TARGET_SUBJECT_KEY}_prediction_plot.png"
    plt.savefig(os.path.join(folder_name, file_name))
    print(f"Prediction plot saved to {os.path.join(folder_name, file_name)}")
    plt.close()
