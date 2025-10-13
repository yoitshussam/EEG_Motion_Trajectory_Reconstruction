# %%
# Install necessary packages
  
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

participant_run_indices = [i for i, _ in enumerate(kin_data) if (i // 9) + 1 == participant_arg]
print(f"Found {len(participant_run_indices)} runs for Participant {participant_arg}.")

num_runs = len(participant_run_indices)
# Calculate split points based on cumulative percentages from the start
train_end_idx = int(num_runs * train_percentage)
val_end_idx = int(num_runs * (train_percentage + val_percentage)) # This is the key change

# Slice the shuffled list of runs to get the indices for each set
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

# print(f"Total pairs generated before split: {len(X_list)}")
# --- Generate windows for each set using the assigned runs ---
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



# --- Convert Data Lists to Single NumPy Arrays ---
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


# --- Convert to PyTorch Tensors and Create DataLoaders ---
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


# Create DataLoaders
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





class   SimpleOptions:
    """A simple class to hold all experiment options,"""
    def __init__(self):
        # ---  Model Parameters ---
        self.model = 'pix2pix'
        self.input_nc = 21
        self.output_nc = 3
        self.ngf = 64
        self.ndf = 64
        self.netG = 'unet_1d'
        self.netD = 'basic_1d'
        self.n_layers_D = 3
        self.num_downs = 5
        # self.disabled_skips=[2]
        # ---  ATTRIBUTES ---
        self.norm = 'batch'         # Normalization type: [batch | instance | none]
        self.init_type = 'normal'   # Network initialization type
        self.init_gain = 0.02       # Initialization gain
        self.no_dropout = True     # Use dropout in the generator or not

        # --- Training Hyperparameters ---
        self.n_epochs = 50
        self.n_epochs_decay = 0
        self.beta1 = 0.5
        self.lr = 0.002
        self.gan_mode = 'vanilla'
        self.pool_size = 0
        self.lr_policy = 'plateau'
        self.lambda_L1 = 100.0

        # --- Experiment and Environment Setup ---
        self.gpu_ids = [0] if torch.cuda.is_available() else []
        self.name = 'eeg_to_kin_pix2pix_1d_notebook'
        self.checkpoints_dir = './checkpoints'
        self.isTrain = True
        self.direction = 'AtoB'
        self.serial_batches = False
        self.num_threads = 0
        self.batch_size = 4
        self.load_size = 256
        self.crop_size = 256
        self.max_dataset_size = float('inf')
        self.preprocess = 'none'
        self.epoch_count = 1
        self.continue_train = False

        # --- Logging/Display Options ---
        self.print_freq = 100
        self.save_epoch_freq = 10000
        self.verbose= False
    


# ---  Get Training Options ---
opt = SimpleOptions()

print("-" * 20)
print(f"DEBUG: About to create model with num_downs = {opt.num_downs}")
print("-" * 20)

save_directory = os.path.join(opt.checkpoints_dir, opt.name)
os.makedirs(save_directory, exist_ok=True)

print(f'The number of training samples = {len(train_loader.dataset)}')
print(f'The number of validation samples = {len(val_loader.dataset)}')

model = create_model(opt)
model.setup(opt)
print(f"Model [{type(model).__name__}] was created successfully.")

# --- Lists to store per-epoch loss values for plotting ---
loss_history = {
    "G_GAN": [], "G_L1": [], "D_real": [], "D_fake": [], "val_G_L1": []
}

total_iters = 0
for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
    epoch_start_time = time.time()
    
    # --- TRAINING ---
    for name in model.model_names:
        if isinstance(name, str):
            net = getattr(model, 'net' + name)
            net.train()

    epoch_train_losses = {"G_GAN": 0.0, "G_L1": 0.0, "D_real": 0.0, "D_fake": 0.0}
    
    for i, data in enumerate(train_loader):
        total_iters += opt.batch_size
        
        # --- CORRECTED DATA HANDLING (for Training Loop) ---
        # 1. Unpack the list from the DataLoader
        eeg_data, kin_data = data
        # 2. Permute kinematics to the expected (N, C, L) shape
        kin_data = kin_data.permute(0, 2, 1).contiguous()
        # 3. Create the dictionary that the model expects
        input_dict = {'A': eeg_data, 'B': kin_data, 'A_paths': '', 'B_paths': ''}

        # 4. Pass the correctly formatted dictionary to the model
        model.set_input(input_dict)
        model.optimize_parameters()

        # Accumulate losses from the current batch
        losses = model.get_current_losses()
        for loss_name, loss_val in losses.items():
            epoch_train_losses[loss_name] += loss_val

    # --- End of Training Epoch ---
    # Calculate and store average training losses
    num_train_batches = len(train_loader)
    for loss_name, accumulated_loss in epoch_train_losses.items():
        loss_history[loss_name].append(accumulated_loss / num_train_batches)

    if epoch % opt.save_epoch_freq == 0:
        model.save_networks('latest')
        model.save_networks(epoch)

    print(f'End of training for epoch {epoch} \t Time Taken: {time.time() - epoch_start_time:.3f} sec')
    model.update_learning_rate()
    
    # --- VALIDATION ---
    model.eval()
    val_l1_loss = 0.0
    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            
            # --- CORRECTED DATA HANDLING (for Validation Loop) ---
            eeg_data, kin_data = val_data
            kin_data = kin_data.permute(0, 2, 1).contiguous()
            input_dict = {'A': eeg_data, 'B': kin_data, 'A_paths': '', 'B_paths': ''}
            
            model.set_input(input_dict)
            model.test()
            val_l1_loss += model.loss_G_L1.item()
            
    avg_val_loss = val_l1_loss / len(val_loader)
    loss_history["val_G_L1"].append(avg_val_loss)
    
    print(f"  [Validation] epoch: {epoch}, Average L1 Loss: {avg_val_loss:.4f}")
    print(f"  [Train]      epoch: {epoch}, G_GAN: {loss_history['G_GAN'][-1]:.4f}, G_L1: {loss_history['G_L1'][-1]:.4f}")
    print("-" * 50)

# --- Plotting ---

print("Training complete. Plotting losses...")

# Create a plot with subplots for different loss types
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
fig.suptitle('Loss History', fontsize=16)
epochs = range(1, len(loss_history['G_L1']) + 1)

# Subplot 1: Generator L1 Loss (Train vs. Validation)
ax1.plot(epochs, loss_history['G_L1'], 'b-', label='Training G_L1 Loss')
ax1.plot(epochs, loss_history['val_G_L1'], 'r-', label='Validation G_L1 Loss')
ax1.set_ylabel('L1 Loss')
ax1.set_title('Generator L1 (Accuracy) Loss')
ax1.legend()
ax1.grid(True, linestyle=':')

# Subplot 2: Adversarial Losses (G_GAN vs. D_real/D_fake)
ax2.plot(epochs, loss_history['G_GAN'], 'g-', label='Generator GAN Loss')
ax2.plot(epochs, loss_history['D_real'], 'c--', label='Discriminator Real Loss', alpha=0.7)
ax2.plot(epochs, loss_history['D_fake'], 'm--', label='Discriminator Fake Loss', alpha=0.7)
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.set_title('Adversarial Losses')
ax2.legend()
ax2.grid(True, linestyle=':')

plt.tight_layout(rect=[0, 0, 1, 0.96])

folder_name=f"GAN_shift_{train_val_step_samples}_lag{lag_s*1000}ms_0.1-40hz"
os.makedirs(folder_name, exist_ok=True)

file_name = f"participant_{participant_arg}_shift_{train_val_step_samples}loss_plots.png"
save_path = os.path.join(folder_name, file_name)

try:
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as {file_name}")
except Exception as e_save:
    print(f"Error saving plot: {e_save}")








print("--- Starting Evaluation on Test Set ---")

# --- 1. Prediction Phase ---
model.eval()
all_predictions = []
all_targets = []

with torch.no_grad():
    for test_data in test_loader:
        eeg_data, kin_data = test_data
        
        input_dict = {
            'A': eeg_data,
            'B': kin_data.permute(0, 2, 1).contiguous(),
            'A_paths': '', 'B_paths': ''
        }
        
        model.set_input(input_dict)
        model.forward()
        
        predictions = model.fake_B.permute(0, 2, 1).cpu().numpy()
        targets = model.real_B.permute(0, 2, 1).cpu().numpy()
        
        all_predictions.append(predictions)
        all_targets.append(targets)

final_predictions = np.concatenate(all_predictions, axis=0)
final_targets = np.concatenate(all_targets, axis=0)

print(f"\nShape of final windowed predictions array: {final_predictions.shape}")
print(f"Shape of final windowed targets array: {final_targets.shape}")


# --- 2. Pearson Correlation Coefficient (PCC) Calculation ---
print("\n--- Calculating Pearson Correlation Coefficient (PCC) ---")
num_features = final_predictions.shape[2]
predictions_flat = final_predictions.reshape(-1, num_features)
targets_flat = final_targets.reshape(-1, num_features)
print(f"Reshaped arrays for PCC to: {predictions_flat.shape}")

pcc_results = {}
pcc_text_lines = []
dimensions = ['X', 'Y', 'Z']
mse_results={}
mse_text_lines=[]
mae_results={}
mae_text_lines=[]

for i, dim_name in enumerate(dimensions):
    pred_dim = predictions_flat[:, i]
    target_dim = targets_flat[:, i]


    mse=sqrt(mean_squared_error(target_dim,pred_dim))
    mae=mean_absolute_error(target_dim,pred_dim)
    corr, p_value = pearsonr(pred_dim, target_dim)
    mse_results[dim_name]=mse
    mae_results[dim_name]=mae
    pcc_results[dim_name] = (corr, p_value)
    mse_text_lines.append(f"RMSE ({dim_name}): {mse:.3f}")
    pcc_text_lines.append(f"PCC ({dim_name}): {corr:.3f}")
    mae_text_lines.append(f"MAE ({dim_name}): {mae:.3f}")

avg_mse=np.mean(list(mse_results.values()))
avg_mae=np.mean(list(mae_results.values()))
mae_text_lines.append(f"Average MAE: {avg_mae:.3f}")

mse_text_lines.append(f"Average RMSE: {avg_mse:.3f}")

valid_corrs = [r[0] for r in pcc_results.values() if not np.isnan(r[0])]
if valid_corrs:
    avg_pcc = np.mean(valid_corrs)
    avg_pcc_text = f"Average PCC: {avg_pcc:.3f}"
    pcc_text_lines.append(avg_pcc_text)
    print(f"\n  {avg_pcc_text}")

with open(f'GAN_lag{lag_s*1000}ms_0.1-40hz.csv', 'a') as fd:
    correlations = [r[0] for r in (list(pcc_results.values()))]
    distances=list(mse_results.values())
    distances_a=list(mae_results.values())

    line_to_write = f"participant:{participant_arg}\nPCC:"+",".join([f"{corr:.3f}" for corr in correlations]) +f",{avg_pcc:.3f}"+"\n"
    line_to_write += f"RMSE:"+",".join([f"{dist:.3f}" for dist in distances]) +f",{avg_mse:.3f}"+"\n"
    line_to_write += f"MAE:"+",".join([f"{dist:.3f}" for dist in distances_a]) +f",{avg_mae:.3f}"+"\n"

    fd.write(line_to_write)


# --- 3. Plotting Results ---
print("\n--- Plotting Continuous Results ---")
duration_to_plot_sec = 20
sampling_rate_hz = 500
continuous_predictions = final_predictions.reshape(-1, num_features)
continuous_targets = final_targets.reshape(-1, num_features)

if continuous_predictions.size > 0:
    total_timesteps = continuous_predictions.shape[0]
    time_axis = np.arange(total_timesteps) / sampling_rate_hz
    timesteps_to_plot = min(total_timesteps, int(duration_to_plot_sec * sampling_rate_hz))
    
    plot_preds = continuous_predictions[:timesteps_to_plot]
    plot_targets = continuous_targets[:timesteps_to_plot]
    plot_time_axis = time_axis[:timesteps_to_plot]
    print(f"Plotting the first {plot_time_axis[-1]:.2f} seconds of the continuous signal.")
    
    fig, axes = plt.subplots(num_features, 1, figsize=(15, num_features * 3), sharex=True, squeeze=False)
    fig.suptitle(f"Continuous Kinematics (First {duration_to_plot_sec}s): Predicted vs. Ground Truth", fontsize=16)
    
    for i in range(num_features):
        ax = axes[i, 0]
        dim_label = dimensions[i]
        ax.plot(plot_time_axis, plot_targets[:, i], label='Ground Truth', color='black', linewidth=1.5)
        ax.plot(plot_time_axis, plot_preds[:, i], label='Predicted', color='red', linestyle='--', linewidth=0.8)
        ax.set_title(f"{dim_label} Trajectory")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle=':')
        ax.set_ylim(0, 1)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    axes[-1, 0].set_xlabel("Time (seconds)")
    
# In your plotting script, at the very end

# --- Add the PCC results text box to the bottom of the figure ---
    mae_full_text="\n".join(mae_text_lines)

    mse_full_text="\n".join(mse_text_lines)
    pcc_full_text = "\n".join(pcc_text_lines)
    bbox_props = dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5)
    fig.text(0.37, 0.01, mae_full_text, ha='center', va='bottom', fontsize=9, bbox=bbox_props)

    fig.text(0.47, 0.01, pcc_full_text, ha='center', va='bottom', fontsize=9, bbox=bbox_props)

    # Place the MSE box on the right side (x=0.7)
    fig.text(0.57, 0.01, mse_full_text, ha='center', va='bottom', fontsize=9, bbox=bbox_props)
    plt.tight_layout(rect=[0, 0.1, 1, 0.96])

    file_name = f"participant_{participant_arg}_shift_{train_val_step_samples}prediction_plots.png"
    save_path = os.path.join(folder_name, file_name)

    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as {file_name}")
    except Exception as e_save:
        print(f"Error saving plot: {e_save}")
else:
    print("No data available to plot.")

