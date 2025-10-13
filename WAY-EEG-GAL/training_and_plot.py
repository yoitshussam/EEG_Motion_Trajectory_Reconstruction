  
import os
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import resample
import scipy.io
import numpy as np
import os
import gc
import mne
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, TensorDataset, DataLoader
import torch.optim as optim
import copy 
from scipy.stats import pearsonr
import sys 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import sys # For command-line arguments
from math import sqrt
print("All modules installed and imported successfully!")


#Command line arguments are 
#1: participant number
#2: sliding window shift sample size

try:
    sys.argv[1]
except NameError:
    participant_arg= 3
else:
    participant_arg= int(sys.argv[1])

# try:
#     sys.argv[2]
# except NameError:
#     eeg_window_step_samples = 50

# else:
#     eeg_window_step_samples = int(sys.argv[2])   # Step for sliding window / kinematic point sampling


  
kin_data = np.load("kin_data_position.npy", allow_pickle=True)
wrist_kin=[]

#these are the indices which correspond to the wrist's x,y,z coordinates
wrist_indices = [2, 5, 8]

for session in kin_data:
    wrist_kin.append(session[:,wrist_indices])


  

# --- Configuration ---
data_root_dir = "." 
eeg_fif_path="unalteredbutfiltered0.5-3.fif"


num_total_participants = 12
sfreq=500
# --- Windowing Parameters ---
window_length_sec = 0.500
window_length_samples = int(window_length_sec * sfreq) # 256
train_val_step_samples = int(round(0.5*(window_length_samples)))  # Overlapping stride
test_step_samples = window_length_samples # Non-overlapping stride

# --- Define Split Percentages for Runs ---
train_percentage = 0.80
val_percentage = 0.10
lag_s=0.250

# --- Load Kinematic Data ---
kin_data = wrist_kin 

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

# Calculate split points based on cumulative percentages from the start
train_end_idx = int(num_runs * train_percentage)
val_end_idx = int(num_runs * (train_percentage + val_percentage)) # This is the key change

# Slice the shuffled list of runs to get the indices for each set
train_run_indices = participant_run_indices[:train_end_idx]
val_run_indices = participant_run_indices[train_end_idx:val_end_idx]
test_run_indices = participant_run_indices[val_end_idx:]

# ... the rest of the script remains the same

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

# %%


# %%


# --- 1. Convert Data Lists to Single NumPy Arrays ---
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

# --- 2. Kinematic Data (y): MinMax Scaling ---
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


# ---Convert to PyTorch Tensors and Create DataLoaders ---
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
batch_size = 500
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
val_loader   = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

print(f"\nDataLoaders created with batch_size={batch_size}.")
print("Setup of Tensors and DataLoaders is complete.")

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





class PremovNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PremovNet, self).__init__()

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
        x = torch.relu(self.conv2(x))  # Apply second convolution
        x = self.maxpool2(x)  # Apply second maxpooling
        x = self.dropout(x)  # Apply dropout
        
        x = x.permute(0, 2, 1)  # Swap back (batch, features, timesteps) → (batch, timesteps, features)
        x, _ = self.lstm(x)  # LSTM expects (batch, timesteps, features)

        x = torch.relu(self.fc1(x))  # Apply fully connected layer to all timesteps
        x = self.fc2(x)  # Output kinematics for each timestep

        return x



    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape (batch, channels, timesteps).

        Returns:
            Tensor: Output tensor of shape (batch, output_dim).
        """
        # Apply BatchNorm
        x = self.batch_norm(x)  # (B, C, T)

        # Conv + Pool 1
        x = torch.relu(self.conv1(x))   # (B, 256, T)
        x = self.pool1(x)               # (B, 256, T//5)

        # Conv + Pool 2
        x = torch.relu(self.conv2(x))   # (B, 128, T//5)
        x = self.pool2(x)               # (B, 128, T//15)

        # Dropout
        x = self.dropout(x)

        # Prepare for LSTM: (B, C, T) → (B, T, C)
        x = x.permute(0, 2, 1)

        # LSTM
        lstm_out, _ = self.lstm(x)      # (B, T, hidden)
        # last_output = lstm_out[:, -1, :]  # take last timestep

        # ReLU after LSTM
        last_output = torch.relu(last_output)

        # Fully Connected Layers
        x = torch.relu(self.fc1(last_output))
        x = self.fc2(x)

        return x





  

os.environ["TORCH_USE_CUDA_DSA"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = 21  # Number of features
output_dim = 3 # Number of target values

time_steps = 500

model=PremovNet(input_dim,output_dim).to(device)


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
        loss = criterion(output, batch_y)

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
            val_loss = criterion(val_output, batch_y_val)
            
            if torch.isnan(val_loss).any() or torch.isinf(val_loss).any():
                 print(f"  WARNING: Invalid loss detected during VALIDATION! Epoch {epoch+1}. Skipping batch.")
                 continue # Skip accumulating invalid loss

            total_val_loss += val_loss.item()
            val_batches +=  1

    # Avoid division by zero if val_loader is empty or all batches had NaN loss
    avg_val_loss = total_val_loss / val_batches if val_batches > 0 else None


    return avg_train_loss, avg_val_loss


# --- Early Stopping Parameters ---
patience = 10  # How many epochs to wait after last improvement
best_val_loss = np.inf # Initialize best validation loss to infinity
epochs_no_improve = 0  # Counter for epochs without improvement
best_model_state = None # To store the state_dict of the best model

# Training loop
num_epochs = 400 # Keep the original maximum epochs
# Ensure 'model' is defined before this print statement
print(f"Starting training for {type(model).__name__} targeting Wrist (max {num_epochs} epochs, patience={patience})...")

for epoch in range(num_epochs):
    # Call the training function for one epoch, passing wrist-specific loaders
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
    save_path = f"best_{type(model).__name__}_wrist_state.pth" 
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

folder_name=f"window{window_length_sec}LAG{lag_s*1000}ms"
os.makedirs(folder_name, exist_ok=True)

file_name = f"participants_{participant_arg}_window{window_length_sec}_training_validation_loss.png"
save_path = os.path.join(folder_name, file_name)

try:
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as {file_name}")
except Exception as e_save:
    print(f"Error saving plot: {e_save}")







print("Starting prediction phase for Wrist model...")
model.eval()  # Set the wrist model to evaluation mode

all_wrist_predictions = []
all_wrist_targets = []

with torch.no_grad():  # Disable gradient calculations
    # Use the correct loader for wrist validation/test data

    for batch_X, batch_y_wrist in test_loader:
        # Move data to the appropriate device
        batch_X, batch_y_wrist = batch_X.to(device), batch_y_wrist.to(device)

        # Get model predictions for wrist
        predictions_wrist = model(batch_X) # Model should output shape (batch, time, 3)

        # Store predictions and targets
        all_wrist_predictions.append(predictions_wrist.cpu().numpy())
        all_wrist_targets.append(batch_y_wrist.cpu().numpy())

print("Wrist prediction phase complete.")

# --- Combine wrist predictions and targets ---
final_predictions = np.concatenate(all_wrist_predictions, axis=0)
final_targets = np.concatenate(all_wrist_targets, axis=0)

print(f"\nShape of final windowed predictions array: {final_predictions.shape}")
print(f"Shape of final windowed targets array: {final_targets.shape}")


# ---  Pearson Correlation Coefficient (PCC) Calculation ---
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

with open(f'metrics_seq2seq_frequencyband_0.5-3hzLAG250ms.csv', 'a') as fd:
    correlations = [r[0] for r in (list(pcc_results.values()))]
    distances=list(mse_results.values())
    distances_a=list(mae_results.values())

    line_to_write = f"participant:{participant_arg}\nPCC:"+",".join([f"{corr:.3f}" for corr in correlations]) +f",{avg_pcc:.3f}"+"\n"
    line_to_write += f"RMSE:"+",".join([f"{dist:.3f}" for dist in distances]) +f",{avg_mse:.3f}"+"\n"
    line_to_write += f"MAE:"+",".join([f"{dist:.3f}" for dist in distances_a]) +f",{avg_mae:.3f}"+"\n"

    fd.write(line_to_write)

# ---  Plotting Results ---
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
    

# --- Add the PCC results text box to the bottom of the figure ---
    mae_full_text="\n".join(mae_text_lines)

    mse_full_text="\n".join(mse_text_lines)
    pcc_full_text = "\n".join(pcc_text_lines)
    bbox_props = dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5)
    fig.text(0.37, 0.01, mae_full_text, ha='center', va='bottom', fontsize=9, bbox=bbox_props)

    fig.text(0.47, 0.01, pcc_full_text, ha='center', va='bottom', fontsize=9, bbox=bbox_props)

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

