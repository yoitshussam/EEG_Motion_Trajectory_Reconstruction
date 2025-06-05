  
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

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import sys # For command-line arguments

print("All modules installed and imported successfully!")


#Command line arguments are 
#1: participant number
#2: sliding window shift sample size
#3: model architecture (1) PreMovNet CNNLSTM, and (2) 2D CNN

try:
    sys.argv[1]
except NameError:
    participant_arg= 3
else:
    participant_arg= int(sys.argv[1])

try:
    sys.argv[2]
except NameError:
    eeg_window_step_samples = 50

else:
    eeg_window_step_samples = int(sys.argv[2])   # Step for sliding window / kinematic point sampling


  
kin_data = np.load("kin_data.npy", allow_pickle=True)
wrist_kin=[]
wrist_indices = [2, 5, 8]

for session in kin_data:
    wrist_kin.append(session[:,wrist_indices])


  

# --- Configuration ---
data_root_dir = "." 
eeg_fif_path = "filtered_clean_eeg_0.53.fif"
num_total_participants = 12

# --- Load Kinematic Data ---
# kinematic data (e.g., wrist features only, shape: (timesteps, 3)).

kin_data = wrist_kin 

# --- Load EEG Data ---
print(f"Loading EEG data from {eeg_fif_path}...")
try:
    eeg_raw_alldata = mne.io.read_raw_fif(eeg_fif_path, preload=False, verbose='WARNING')
except FileNotFoundError:
    print(f"ERROR: EEG file not found at {eeg_fif_path}")
    exit()
except Exception as e:
    print(f"ERROR: Could not load EEG data. {e}")
    exit()

# --- Define Common Sampling Frequency ---
sfreq = 500  # Hz
print(f"Using common sampling frequency: {sfreq} Hz for EEG and Kinematics.")

# --- Calculate EEG session start points based on kinematic session lengths ---
# This assumes 'kin_data' (i.e., 'wrist_kin') is a list of sessions,
# and each session's length corresponds to a contiguous block in the EEG .fif file.
try:
    kin_session_lengths_timesteps = [session.shape[0] for session in kin_data]
except AttributeError:
    print("ERROR: 'kin_data' (wrist_kin) does not seem to be a list of NumPy arrays. "
          "Each element should have a .shape attribute.")
    exit()
    
eeg_session_lengths_samples = kin_session_lengths_timesteps
eeg_session_start_indices = [0] + list(np.cumsum(eeg_session_lengths_samples[:-1]))

# --- Participant and Run Mapping Logic Parameters ---
current_participant_id_loaded = -1
p_lifts_matrix = None 

# Define EEG window parameters relative to EACH kinematic point to be predicted
# e.g., EEG window from 350ms before kin_point to 50ms before kin_point
eeg_window_start_offset_sec = -0.5 # Start of EEG window relative to kin_point
eeg_window_end_offset_sec = 0    # End of EEG window relative to kin_point
# eeg_window_step_samples = 10         # Step for sliding window / kinematic point sampling

# Initialize lists to store the processed data
eeg_data_new = []
kin_data_new = [] # This will store individual kinematic points

session_cumsum = 0 #

# Iterate through each session using its global index
# len(kin_data) is the total number of sessions
for global_session_idx in range(len(kin_data)):
    participant_id = (global_session_idx // 9) + 1
    run_number = (global_session_idx % 9) + 1
    if participant_id==participant_arg:
    
        # Load P.AllLifts for the current participant if it's new or changed
        if participant_id != current_participant_id_loaded:
            if p_lifts_matrix is not None:
                del mat_data
                del p_lifts_matrix
                gc.collect()
            
            participant_folder = os.path.join(data_root_dir, f"P{participant_id}")
            all_lifts_path = os.path.join(participant_folder, f"P{participant_id}_AllLifts.mat")
            
            try:
                mat_data = scipy.io.loadmat(all_lifts_path)
                p_lifts_matrix = mat_data['P']['AllLifts'][0,0]
                current_participant_id_loaded = participant_id
                print(f"  Loaded P{participant_id}_AllLifts.mat")
            except FileNotFoundError:
                print(f"  ERROR: P{participant_id}_AllLifts.mat not found. Skipping participant {participant_id}.")
                p_lifts_matrix = None # Ensure it's None so subsequent runs for this P are skipped
                continue 
            except Exception as e:
                print(f"  ERROR: Could not load P{participant_id}_AllLifts.mat: {e}. Skipping.")
                p_lifts_matrix = None
                continue

        if p_lifts_matrix is None:
            print(f"  P.AllLifts matrix not available for P{participant_id}. Skipping run {run_number}.")
            continue

        current_run_kin_data = kin_data[global_session_idx] # Kinematic data for the current session
        # global start sample of the current session's data within the continuous EEG .fif file
        current_session_global_eeg_start_sample = eeg_session_start_indices[global_session_idx] 

        # Filter p_lifts_matrix for the current participant_id and run_number
        try:
            run_trials_data = p_lifts_matrix[
                (p_lifts_matrix[:, 0] == participant_id) & (p_lifts_matrix[:, 1] == run_number)
            ]
        except IndexError as e:
            print(f"  ERROR: Indexing p_lifts_matrix for P{participant_id}R{run_number}. Shape: {p_lifts_matrix.shape}. Error: {e}")
            continue

        session_cumsum += run_trials_data.shape[0]
        print(f"  Found {run_trials_data.shape[0]} trials (lifts) for P{participant_id}R{run_number}.")

        # Iterate through each trial (lift) in the current session
        for trial_row_idx in range(run_trials_data.shape[0]):
            trial_row = run_trials_data[trial_row_idx, :]
            lift_number = int(trial_row[2])
            start_time_in_run_sec = trial_row[7]    # StartTime of lift relative to run start
            thandstart_relative_sec = trial_row[33] # tHandStart relative to lift StartTime
            thandstop_relative_sec = trial_row[34]  # tHandStop relative to lift StartTime
            # print(f"hand time= {(thandstop_relative_sec-thandstart_relative_sec)}")
            # Absolute time of hand movement start/stop from the beginning of the current run's kinematic data
            abs_thandstart_sec_in_run = start_time_in_run_sec + thandstart_relative_sec
            abs_thandstop_sec_in_run = start_time_in_run_sec + thandstop_relative_sec

            # Convert these times to sample indices within the current run's kinematic data array
            kin_lift_start_sample_in_run = int(round(abs_thandstart_sec_in_run * sfreq))
            kin_lift_end_sample_in_run = int(round(abs_thandstop_sec_in_run * sfreq))

            # Iterate through the kinematic data of the current lift with a sliding window
            # kin_target_idx_in_run is an index relative to the start of current_run_kin_data
            for kin_target_idx_in_run in range(kin_lift_start_sample_in_run, kin_lift_end_sample_in_run + 1, eeg_window_step_samples):
                
                # Ensure the target kinematic index is within the bounds of the current session's kinematic data
                if not (0 <= kin_target_idx_in_run < current_run_kin_data.shape[0]):
                    # print(f"  Skipping kin_target_idx_in_run {kin_target_idx_in_run} for P{participant_id}R{run_number}L{lift_number} (out of bounds for session kin data: {current_run_kin_data.shape[0]})")
                    continue

                # 1. Get the target kinematic point (e.g., [x, y, z] for wrist)
                target_kin_point = current_run_kin_data[kin_target_idx_in_run, :]

                # 2. Determine the EEG window preceding this kinematic point
                # Global sample index of the target kinematic point in the continuous EEG .fif file
                global_kin_target_sample_in_fif = current_session_global_eeg_start_sample + kin_target_idx_in_run
                
                # Calculate EEG window start and end samples in the global EEG .fif file
                eeg_window_start_global = global_kin_target_sample_in_fif + int(round(eeg_window_start_offset_sec * sfreq))
                eeg_window_end_global = global_kin_target_sample_in_fif + int(round(eeg_window_end_offset_sec * sfreq))

                # 3. Extract the EEG segment
                if eeg_window_start_global >= eeg_window_end_global:
                    # print(f"  Skipping due to invalid EEG window (start >= end) for P{participant_id}R{run_number}L{lift_number} at kin_idx {kin_target_idx_in_run}")
                    continue
                if eeg_window_start_global < 0 or eeg_window_end_global > eeg_raw_alldata.n_times:
                    # print(f"  Skipping due to EEG window out of bounds for P{participant_id}R{run_number}L{lift_number} at kin_idx {kin_target_idx_in_run}")
                    continue

                current_eeg_segment, _ = eeg_raw_alldata[:, eeg_window_start_global:eeg_window_end_global]

                # 4. Store if EEG segment is valid
                if current_eeg_segment.shape[1] > 0: # Check if EEG segment is not empty
                    eeg_data_new.append(current_eeg_segment.copy())
                    kin_data_new.append(target_kin_point.copy()) # target_kin_point is already a single point (1D array)
                # else:
                    # print(f"  Skipping due to empty EEG segment for P{participant_id}R{run_number}L{lift_number} at kin_idx {kin_target_idx_in_run}")


# Clean up
if p_lifts_matrix is not None:
    if 'mat_data' in locals() or 'mat_data' in globals():
        del mat_data
    del p_lifts_matrix
    gc.collect()

# Note: all_trial_processed_data was not used for appending in this new logic.
print(f"\nProcessed {len(eeg_data_new)} EEG-Kinematic pairs.")

# Now, eeg_data_new is a list of EEG segments (e.g., shape (channels, 150))
# and kin_data_new is a list of corresponding single kinematic points (e.g., shape (3,) for wrist x,y,z)




  


# 1. Get the total number of EEG-Kinematic pairs
total_samples = len(eeg_data_new)
print(f"Total EEG-Kinematic pairs available for splitting: {total_samples}")

# 2. Define your desired split percentages
train_percentage = 0.80  # 80% for training
val_percentage = 0.10    # 10% for validation
# Test percentage will be the remainder (1.0 - train_percentage - val_percentage)

num_train_samples = int(total_samples * train_percentage)
num_val_samples = int(total_samples * val_percentage)
num_test_samples = total_samples - num_train_samples - num_val_samples

# 4. Calculate the end indices for slicing
# Ensure indices are integers
train_end_idx = num_train_samples
val_end_idx = num_train_samples + num_val_samples
# test_end_idx is implicitly total_samples, so slicing up to the end works


# 5. Perform the split

X_train = eeg_data_new[0 : train_end_idx]
y_train = kin_data_new[0 : train_end_idx]

# Validation Set
X_val = eeg_data_new[train_end_idx : val_end_idx]
y_val = kin_data_new[train_end_idx : val_end_idx]

# Test Set
X_test = eeg_data_new[val_end_idx : total_samples] # Slicing up to total_samples
y_test = kin_data_new[val_end_idx : total_samples]

print("\nData splitting complete.")
print(f"  Length of X_train: {len(X_train)}, y_train: {len(y_train)}")
print(f"  Length of X_val:   {len(X_val)}, y_val: {len(y_val)}")
print(f"  Length of X_test:  {len(X_test)}, y_test: {len(y_test)}")



  
#Normalization 



# --- 1. Kinematic Data (y_train, y_val, y_test): MinMax Scaling ---
print("Normalizing Kinematic data (MinMaxScaler)...")


y_train_combined_2d = np.array(y_train)

if y_train_combined_2d.shape[0] > 0:
    scaler_kin = MinMaxScaler(feature_range=(0, 1))
    scaler_kin.fit(y_train_combined_2d)

    def transform_kin_point(point_1d, scaler):
        if isinstance(point_1d, np.ndarray) and point_1d.ndim == 1 and point_1d.shape[0] > 0:
            point_2d = point_1d.reshape(1, -1)
            transformed_point_2d = scaler.transform(point_2d)
            return transformed_point_2d.flatten()

    # Reassign to original variables
    y_train = [transform_kin_point(point, scaler_kin) for point in y_train]
    y_val   = [transform_kin_point(point, scaler_kin) for point in y_val]
    y_test  = [transform_kin_point(point, scaler_kin) for point in y_test]
    
    print("Kinematic data normalized and reassigned.")
else:
    print("WARNING: Combined kinematic training data (after stacking) is empty. Skipping kinematic normalization.")


# --- 2. EEG Data (X_train, X_val, X_test): Z-Score Normalization (Per Channel) ---
print("\nNormalizing EEG data (Z-Score per channel)...")

X_train_transposed = []
for segment in X_train:
        X_train_transposed.append(segment.T) 

if not X_train_transposed:
    print("WARNING: No valid EEG training data to compute mean/std. Skipping EEG normalization.")
else:
    X_train_combined_transposed = np.concatenate(X_train_transposed, axis=0)

    if X_train_combined_transposed.shape[0] > 0:
        eeg_mean_per_channel = np.mean(X_train_combined_transposed, axis=0) 
        eeg_std_per_channel = np.std(X_train_combined_transposed, axis=0)   

        epsilon = 1e-8
        eeg_std_safe_per_channel = eeg_std_per_channel + epsilon

        def z_score_eeg_segment(segment_original_shape, mean_vals, std_vals_safe):
            segment_T = segment_original_shape.T 
            normalized_segment_T = (segment_T - mean_vals) / std_vals_safe
            return normalized_segment_T.T 

        # Reassign to original variables
        X_train = [z_score_eeg_segment(segment, eeg_mean_per_channel, eeg_std_safe_per_channel) for segment in X_train]
        X_val   = [z_score_eeg_segment(segment, eeg_mean_per_channel, eeg_std_safe_per_channel) for segment in X_val]
        X_test  = [z_score_eeg_segment(segment, eeg_mean_per_channel, eeg_std_safe_per_channel) for segment in X_test]
        
        print("EEG data normalized and reassigned.")
    else:
        print("WARNING: Combined EEG training data (transposed) is empty. Skipping EEG normalization.")

print("\nNormalization process finished.")

# --- 3. Convert to PyTorch Tensors and Create DataLoaders ---
print("\nConverting data to PyTorch Tensors and creating DataLoaders...")

# Ensure all data items are valid NumPy arrays before converting to Tensors

# Convert lists of NumPy arrays to single large PyTorch Tensors
# EEG data: (num_samples, channels, eeg_timesteps)
# Kinematic data: (num_samples, kin_features)

# Stack along a new dimension (sample dimension)
X_train_tensor = torch.from_numpy(np.array(X_train, dtype=np.float32))
y_train_tensor = torch.from_numpy(np.array(y_train, dtype=np.float32))

X_val_tensor = torch.from_numpy(np.array(X_val, dtype=np.float32))
y_val_tensor = torch.from_numpy(np.array(y_val, dtype=np.float32))

X_test_tensor = torch.from_numpy(np.array(X_test, dtype=np.float32))
y_test_tensor = torch.from_numpy(np.array(y_test, dtype=np.float32))

print(f"Shape of X_train_tensor: {X_train_tensor.shape}")
print(f"Shape of y_train_tensor: {y_train_tensor.shape}")

# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset   = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset  = TensorDataset(X_test_tensor, y_test_tensor)

print(f"\nTrain dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")



  
# Create DataLoaders
batch_size = 256
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
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
    def __init__(self, input_dim, output_dim, lstm_hidden_size=128, lstm_layers=1):
        """
        PremovNet for sequence-to-point prediction.
        Predicts the output corresponding to the last time step of the input sequence.

        Args:
            input_dim (int): Number of input features (EEG channels).
            output_dim (int): Number of output features (e.g., 3 for X,Y,Z of one body part).
            lstm_hidden_size (int): Number of features in the LSTM hidden state.
            lstm_layers (int): Number of recurrent LSTM layers.
        """
        super(PremovNet, self).__init__()
        self.input_dim = input_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers

        # Input Batch Normalization (applied on feature dimension)
        self.batch_norm = nn.BatchNorm1d(input_dim)

        # CNN Layers (padding for 'same' output length with stride=1)
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool1d(kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.dropout_cnn = nn.Dropout(0.5) # Dropout after CNN

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=128, # Input features from CNN
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=True) # Expects (batch, seq_len, features)

        # Fully Connected Layers (process only the final LSTM output)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc1 = nn.Linear(lstm_hidden_size*2, 128) # Takes final LSTM hidden state
        self.fc2 = nn.Linear(128, output_dim)      # Outputs the final prediction

    def forward(self, x):
        """
        Forward pass for sequence-to-point prediction.

        Args:
            x (Tensor): Input tensor of shape (batch, timesteps, features).

        Returns:
            Tensor: Output tensor of shape (batch, output_dim), representing the
                    prediction corresponding to the last input timestep.
        """
        # Input shape: (batch, timesteps, features) e.g., (B, T, C_in)

        # Permute for Conv1d: (B, T, C_in) -> (B, C_in, T)

        # Apply BatchNorm across features
        x = self.batch_norm(x)

        # CNN Feature Extraction
        x = torch.relu(self.conv1(x)) # Shape: (B, 256, T)
        x = self.maxpool1(x)          # Shape: (B, 256, T)
        x = torch.relu(self.conv2(x)) # Shape: (B, 128, T)
        x = self.maxpool2(x)          # Shape: (B, 128, T)
        x = self.dropout_cnn(x)       # Shape: (B, 128, T)

        # Permute for LSTM: (B, 128, T) -> (B, T, 128)
        x = x.permute(0, 2, 1)

        # LSTM Layer
        # lstm_out shape: (batch, timesteps, hidden_size)
        # hidden_state is tuple (h_n, c_n), where h_n is (num_layers, batch, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Select the output from the *last* time step
        # lstm_out shape is (batch, seq_len, hidden_size)
        last_time_step_output = lstm_out[:, -1, :] # Shape: (batch, hidden_size)

        # Pass the last time step's output through FC layers
        x = self.dropout_fc(last_time_step_output)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x) # Final output shape: (batch, output_dim)

        return x


  
class PyTorchCNN(nn.Module):
    def __init__(self, input_height=21, input_width=500, num_output_features=3):
        super(PyTorchCNN, self).__init__()
        # Expects input to forward() to be (batch_size, input_height, input_width)
        # e.g., (batch_size, 21, 500)
        
        # Layer 1: Conv2d + BatchNorm + ReLU
        # Input after unsqueeze in forward(): (batch_size, 1, input_height, input_width)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), padding=(3, 3)) # 'same' padding
        self.bn1 = nn.BatchNorm2d(64)
        
        # Layer 2: Conv2d + BatchNorm + ReLU
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), padding=(2, 2)) # 'same' padding
        self.bn2 = nn.BatchNorm2d(64)
        
        # Layer 3: Conv2d + BatchNorm + ReLU
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1)) # 'same' padding
        self.bn3 = nn.BatchNorm2d(64)
        
        self.flatten = nn.Flatten()
        
        # Calculate flattened features:
        # With 'same' padding, output H, W from conv3 remain input_height, input_width
        flattened_features = 64 * input_height * input_width # e.g., 64 * 21 * 500 = 672,000
        
        # Layer 4: Dense + BatchNorm + ReLU
        self.fc1 = nn.Linear(flattened_features, 32)
        self.bn_fc1 = nn.BatchNorm1d(32)
        
        # Layer 5: Output Layer (3 features)
        self.fc2 = nn.Linear(32, num_output_features)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # x is expected to be (batch_size, height, width), e.g., (N, 21, 500)
        # Add channel dimension for Conv2d: (N, 1, height, width)
        x = x.unsqueeze(1) 
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        x = self.flatten(x) # Shape becomes (N, 64 * height * width)
        
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.fc2(x) # Linear output for MSE loss
        return x



  

os.environ["TORCH_USE_CUDA_DSA"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = 21  # Number of features
output_dim = 3 # Number of target values

time_steps = 500

try:
    sys.argv[3]
except NameError:
    model = PremovNet(input_dim,output_dim).to(device)
else:
    if(sys.argv[3]==2):
        model= PyTorchCNN.to(device)        
    else:
        model = PremovNet(input_dim,output_dim).to(device)



criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-2)
print(input_dim)
print(output_dim)

  


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy # Needed for saving best model state


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


# =============================================
# -------------------------------------------------------------------


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
    save_path = f"best_{type(model).__name__}_wrist_state.pth" # Example filename for wrist model
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

folder_name=f"shift_{eeg_window_step_samples}_time_plot_{eeg_window_start_offset_sec*-1*1000}ms_window"
os.makedirs(folder_name, exist_ok=True)

file_name = f"participants_{participant_arg}_shift_{eeg_window_step_samples}_training_validation_loss_{eeg_window_start_offset_sec*-1*1000}ms_window.png"
save_path = os.path.join(folder_name, file_name)

try:
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as {file_name}")
except Exception as e_save:
    print(f"Error saving plot: {e_save}")






# --- Model with best weights is ready ---
# You can now use 'model' (which now holds the best wrist weights) for testing or prediction

print("Starting prediction phase for Wrist model...")
model.eval()  # Set the wrist model to evaluation mode

all_wrist_predictions = []
all_wrist_targets = []

with torch.no_grad():  # Disable gradient calculations
    # Use the correct loader for wrist validation/test data
    data_loader_for_predictions = val_loader # Or test_loader_wrist

    for batch_X, batch_y_wrist in data_loader_for_predictions:
        # Move data to the appropriate device
        batch_X, batch_y_wrist = batch_X.to(device), batch_y_wrist.to(device)

        # Get model predictions for wrist
        predictions_wrist = model(batch_X) # Model should output shape (batch, time, 3)

        # Store predictions and targets
        all_wrist_predictions.append(predictions_wrist.cpu().numpy())
        all_wrist_targets.append(batch_y_wrist.cpu().numpy())

print("Wrist prediction phase complete.")

# --- Combine wrist predictions and targets ---
final_wrist_predictions = np.concatenate(all_wrist_predictions, axis=0)
final_wrist_targets = np.concatenate(all_wrist_targets, axis=0)

print(f"\nShape of final WRIST predictions array: {final_wrist_predictions.shape}")
# Expected shape: (total_val_windows, window_size, 3)
print(f"Shape of final WRIST targets array: {final_wrist_targets.shape}")
# Expected shape: (total_val_windows, window_size, 3)


  
if not isinstance(final_wrist_predictions, np.ndarray):
    final_wrist_predictions = np.array(final_wrist_predictions)
if 'final_wrist_targets' in locals() and not isinstance(final_wrist_targets, np.ndarray):
    final_wrist_targets = np.array(final_wrist_targets)

is_single_sequence_input = False
if final_wrist_predictions.ndim == 2:
    print(f"Input 'final_wrist_predictions' is 2D {final_wrist_predictions.shape}. "
            f"Assuming single sequence (timesteps, features), reshaping to 3D.")
    final_wrist_predictions = np.expand_dims(final_wrist_predictions, axis=0)
    if 'final_wrist_targets' in locals() and final_wrist_targets.ndim == 2:
            final_wrist_targets = np.expand_dims(final_wrist_targets, axis=0)
    is_single_sequence_input = True
elif final_wrist_predictions.ndim != 3:
    raise ValueError(f"Expected final_wrist_predictions to be 2D or 3D, "
                        f"got {final_wrist_predictions.ndim}D shape {final_wrist_predictions.shape}")


# --- Assume the following NumPy arrays exist from the prediction step ---
# final_wrist_predictions: shape (total_windows, window_size, 3)
# final_wrist_targets:   shape (total_windows, window_size, 3)
# where the last dimension is (X, Y, Z)

print("--- Calculating Pearson Correlation Coefficient (PCC) for Wrist ---")

# --- Input Data Validation ---

    # --- Reshape data ---
    # Combine the window and time dimensions to get (total_timesteps, 3)
    # This allows calculating correlation across the entire sequence
num_features = final_wrist_predictions.shape[2]
predictions_flat = final_wrist_predictions.reshape(-1, num_features)
targets_flat = final_wrist_targets.reshape(-1, num_features)
print(f"Reshaped arrays to: {predictions_flat.shape}")

# --- Calculate PCC for each dimension ---
results = {}
dimensions = ['X', 'Y', 'Z']

for i, dim_name in enumerate(dimensions):
    # Extract the column for the current dimension
    pred_dim = predictions_flat[:, i]
    target_dim = targets_flat[:, i]

    # Calculate Pearson correlation coefficient and p-value
    # Check for constant input which leads to NaN correlation
    if np.std(pred_dim) == 0 or np.std(target_dim) == 0:
            print(f"  Skipping PCC for {dim_name}-dimension: Data is constant.")
            results[dim_name] = (np.nan, np.nan) # Store NaN if constant
    else:
        try:
            corr, p_value = pearsonr(pred_dim, target_dim)
            results[dim_name] = (corr, p_value)
            print(f"  PCC ({dim_name}): {corr:.4f} (p-value: {p_value:.2e})")
        except ValueError as e:
                print(f"  Error calculating PCC for {dim_name}-dimension: {e}")
                results[dim_name] = (np.nan, np.nan) # Store NaN on error


    # --- Optional: Calculate average PCC (use with caution) ---
    valid_corrs = [r[0] for r in results.values() if not np.isnan(r[0])]
    if valid_corrs:
        avg_pcc = np.mean(valid_corrs)
        print(f"\n  Average PCC across dimensions: {avg_pcc:.4f}")
    else:
        print("\n  Could not calculate average PCC.")



  


# --- Configuration ---
num_sequences_to_plot = 1 # Plot the first N sequences.
# Set a limit for time to plot in seconds. Set to None to plot the entire duration.
max_time_to_plot_seconds = 5 # Example: Plot up to 10.0 seconds. None to plot all.

plot_main_title_base = f"Wrist Kinematics: Predicted vs. Ground Truth ({eeg_window_start_offset_sec*-1*1000}ms Windows - {(eeg_window_step_samples/500)*1000}ms Step Size)"

# --- Define Sampling Rate and ASSUME Window Step is pre-defined ---
sampling_rate_hz = 500  # EEG sampling rate in Hz


time_per_timestep = eeg_window_step_samples / sampling_rate_hz
if time_per_timestep <= 0:
    raise ValueError("Calculated 'time_per_timestep' must be positive. "
                     "Check 'eeg_window_step_samples' and 'sampling_rate_hz'.")


# --- PCC Calculation ---
print("\n--- Calculating Pearson Correlation Coefficient (PCC) ---")
pcc_results_text_lines = []
avg_pcc_text = "Average PCC: N/A"

if 'final_wrist_predictions' in locals() and 'final_wrist_targets' in locals() and \
   isinstance(final_wrist_predictions, np.ndarray) and isinstance(final_wrist_targets, np.ndarray):

    temp_preds_for_pcc = final_wrist_predictions
    temp_targets_for_pcc = final_wrist_targets

    if temp_preds_for_pcc.ndim == 3:
        num_features_pcc = temp_preds_for_pcc.shape[2]
        predictions_flat = temp_preds_for_pcc.reshape(-1, num_features_pcc)
        targets_flat = temp_targets_for_pcc.reshape(-1, num_features_pcc)
    elif temp_preds_for_pcc.ndim == 2:
        num_features_pcc = temp_preds_for_pcc.shape[1]
        predictions_flat = temp_preds_for_pcc
        targets_flat = temp_targets_for_pcc
    else:
        print("Error: PCC calculation expects 2D or 3D array for predictions and targets.")
        predictions_flat, targets_flat = None, None

    if predictions_flat is not None and targets_flat is not None and predictions_flat.shape == targets_flat.shape:
        if predictions_flat.shape[0] == 0: # No data to process
            pcc_results_text_lines.append("PCC calculation skipped: No data points.")
        else:
            print(f"Reshaped arrays for PCC to: {predictions_flat.shape}")
            pcc_results = {}
            dimensions = ['X', 'Y', 'Z'] + [f"Feature {i+1}" for i in range(3, num_features_pcc)]

            for i in range(num_features_pcc):
                dim_name = dimensions[i] if i < len(dimensions) else f"Feature {i+1}"
                pred_dim = predictions_flat[:, i]
                target_dim = targets_flat[:, i]

                if np.std(pred_dim) < 1e-9 or np.std(target_dim) < 1e-9:
                    print(f"   Skipping PCC for {dim_name}-dimension: Data is effectively constant.")
                    pcc_results[dim_name] = (np.nan, np.nan)
                else:
                    try:
                        corr, p_value = pearsonr(pred_dim, target_dim)
                        pcc_results[dim_name] = (corr, p_value)
                        pcc_results_text_lines.append(f"PCC ({dim_name}): {corr:.3f} (p={p_value:.2e})")
                    except ValueError as e:
                        print(f"   Error calculating PCC for {dim_name}-dimension: {e}")
                        pcc_results[dim_name] = (np.nan, np.nan)

            valid_corrs = [r[0] for r in pcc_results.values() if not np.isnan(r[0])]
            if valid_corrs:
                avg_pcc = np.mean(valid_corrs)
                avg_pcc_text = f"Average PCC: {avg_pcc:.3f}"
            pcc_results_text_lines.append(avg_pcc_text)
    else:
        pcc_results_text_lines.append("PCC calculation skipped due to data shape issues or missing data.")
else:
    pcc_results_text_lines.append("PCC not calculated: prediction/target data missing or not ndarray.")

# --- Plotting ---
print(f"\nPreparing to plot...")
try:
    if not isinstance(final_wrist_predictions, np.ndarray):
        final_wrist_predictions = np.array(final_wrist_predictions)
    if 'final_wrist_targets' in locals() and not isinstance(final_wrist_targets, np.ndarray):
        final_wrist_targets = np.array(final_wrist_targets)

    is_single_sequence_input = False
    if final_wrist_predictions.ndim == 2:
        final_wrist_predictions = np.expand_dims(final_wrist_predictions, axis=0)
        if 'final_wrist_targets' in locals() and final_wrist_targets.ndim == 2:
             final_wrist_targets = np.expand_dims(final_wrist_targets, axis=0)
        is_single_sequence_input = True
    elif final_wrist_predictions.ndim != 3:
        raise ValueError(f"Plotting expects 2D or 3D predictions, got {final_wrist_predictions.ndim}D")

    if 'final_wrist_targets' not in locals():
        raise ValueError("'final_wrist_targets' is not defined.")
    if final_wrist_targets.shape != final_wrist_predictions.shape:
        raise ValueError(f"Shape mismatch for plotting. Pred: {final_wrist_predictions.shape}, Target: {final_wrist_targets.shape}")

    actual_num_sequences_in_data = final_wrist_predictions.shape[0]
    original_sequence_length_timesteps = final_wrist_predictions.shape[1]
    num_features_plot = final_wrist_predictions.shape[2]

    current_num_sequences_to_plot = min(num_sequences_to_plot, actual_num_sequences_in_data)

    if current_num_sequences_to_plot <= 0:
        print("No sequences to plot. Exiting.")
    elif original_sequence_length_timesteps == 0:
        print("Data contains 0 timesteps. Nothing to plot.")
    else:

        plot_dimension_labels = ['X', 'Y', 'Z'] + [f"Feat {i+1}" for i in range(3, num_features_plot)]

        # Determine the number of timesteps to plot based on max_time_to_plot_seconds
        timesteps_to_plot_limit = original_sequence_length_timesteps # Default to all
        if max_time_to_plot_seconds is not None:
            if max_time_to_plot_seconds < 0:
                 print("Warning: 'max_time_to_plot_seconds' is negative. Plotting entire duration.")
            else:
                # Calculate timesteps corresponding to the time limit
                # Using ceil to include the full timestep interval that max_time_to_plot_seconds falls into
                timesteps_requested_by_time_limit = int(np.ceil(max_time_to_plot_seconds / time_per_timestep))
                timesteps_to_plot_limit = min(original_sequence_length_timesteps, timesteps_requested_by_time_limit)

        current_sequence_length_to_plot_timesteps = timesteps_to_plot_limit
        if current_sequence_length_to_plot_timesteps <= 0 and max_time_to_plot_seconds is not None and max_time_to_plot_seconds > 0 :
            print(f"Warning: max_time_to_plot_seconds ({max_time_to_plot_seconds}s) is shorter than one timestep ({time_per_timestep}s). Effective timesteps to plot is 0 or 1.")
            if current_sequence_length_to_plot_timesteps == 0 and original_sequence_length_timesteps > 0 : # Ensure at least one if possible
                 current_sequence_length_to_plot_timesteps = 1 # Plot at least the first timestep if data exists
        elif current_sequence_length_to_plot_timesteps == 0 and original_sequence_length_timesteps > 0:
             print("Warning: Calculated timesteps to plot is 0, but data exists. Plotting first timestep.")
             current_sequence_length_to_plot_timesteps = 1


        actual_plotted_duration_seconds = current_sequence_length_to_plot_timesteps * time_per_timestep
        plot_title_suffix = ""

        if max_time_to_plot_seconds is not None and max_time_to_plot_seconds >= 0:
            if current_sequence_length_to_plot_timesteps < original_sequence_length_timesteps:
                plot_title_suffix = f" (First {actual_plotted_duration_seconds:.2f}s / {current_sequence_length_to_plot_timesteps} Timesteps)"
                print(f"Plotting first {actual_plotted_duration_seconds:.2f}s ({current_sequence_length_to_plot_timesteps} of {original_sequence_length_timesteps} timesteps) due to time limit of {max_time_to_plot_seconds}s.")
            else: # Time limit was set but it was >= data duration or resulted in full plot
                plot_title_suffix = f" (Full {actual_plotted_duration_seconds:.2f}s / {current_sequence_length_to_plot_timesteps} Timesteps)"
                print(f"Plotting all {current_sequence_length_to_plot_timesteps} timesteps (total duration {actual_plotted_duration_seconds:.2f}s). Requested time limit was {max_time_to_plot_seconds}s.")
        else: # No time limit or negative time limit (interpreted as plot all)
            plot_title_suffix = f" (Full {actual_plotted_duration_seconds:.2f}s / {current_sequence_length_to_plot_timesteps} Timesteps)"
            print(f"Plotting all {current_sequence_length_to_plot_timesteps} timesteps (total duration {actual_plotted_duration_seconds:.2f}s).")

        if current_sequence_length_to_plot_timesteps == 0:
            print("Final number of timesteps to plot is zero. No plot will be generated.")
        else:
            num_plot_rows = current_num_sequences_to_plot * num_features_plot
            fig_width = 10
            fig_height_per_feature = 2.5
            fig_height = fig_height_per_feature * num_plot_rows + 1.5

            plt.style.use('seaborn-v0_8-whitegrid')
            fig, axes = plt.subplots(num_plot_rows, 1, figsize=(fig_width, fig_height), sharex=True, squeeze=False)

            main_title = plot_main_title_base + plot_title_suffix
            fig.suptitle(main_title, fontsize=16, y=0.98 if num_plot_rows > 1 else 0.96)

            time_axis_seconds = np.arange(current_sequence_length_to_plot_timesteps) * time_per_timestep
            plot_idx_counter = 0

            for seq_iter_idx in range(current_num_sequences_to_plot):
                seq_preds = final_wrist_predictions[seq_iter_idx, :current_sequence_length_to_plot_timesteps, :]
                seq_targets = final_wrist_targets[seq_iter_idx, :current_sequence_length_to_plot_timesteps, :]

                for feat_idx in range(num_features_plot):
                    ax = axes[plot_idx_counter, 0]
                    dim_label = plot_dimension_labels[feat_idx] if feat_idx < len(plot_dimension_labels) else f"Feat {feat_idx+1}"

                    ax.plot(time_axis_seconds, seq_targets[:, feat_idx], label='Ground Truth', color='black', linewidth=1.2, alpha=0.7)
                    ax.plot(time_axis_seconds, seq_preds[:, feat_idx], label='Predicted', color='red', linestyle='--', linewidth=1.2, alpha=0.7)
                    ax.tick_params(axis='both', which='major', labelsize=8)
                    ax.grid(True, linestyle=':', alpha=0.6)

                    combined_data = np.concatenate((seq_targets[:, feat_idx], seq_preds[:, feat_idx]))
                    y_min, y_max = np.min(combined_data), np.max(combined_data)
                    padding = (y_max - y_min) * 0.1 if (y_max - y_min) > 1e-6 else 0.1
                    ax.set_ylim(y_min - padding, y_max + padding)

                    title_prefix = f"Seq {seq_iter_idx}: " if current_num_sequences_to_plot > 1 else ""
                    ax.set_title(f"{title_prefix}{dim_label} Trajectory", fontsize=10)
                    ax.set_ylabel("Value", fontsize=9)

                    if plot_idx_counter == 0: handles, labels = ax.get_legend_handles_labels()
                    plot_idx_counter += 1

            if plot_idx_counter > 0:
                axes[plot_idx_counter-1, 0].set_xlabel("Time (seconds)", fontsize=9)

            if 'handles' in locals():
                fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.95 if num_plot_rows > 1 else 0.90), fontsize=9)

            pcc_full_text = "PCC Results:\n" + "\n".join(pcc_results_text_lines)
            fig.text(0.5, 0.01, pcc_full_text, ha='center', va='bottom', fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

            plt.tight_layout(rect=[0, 0.05, 1, 0.93 if num_plot_rows > 1 else 0.88])

            file_name = f"participants_{participant_arg}_shift_{eeg_window_step_samples}_time_plot_{eeg_window_start_offset_sec*-1*1000}ms_window.png"
            try:
                save_path = os.path.join(folder_name, file_name)
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                
                print(f"\nPlot saved as {save_path}")
            except Exception as e_save:
                print(f"Error saving plot: {e_save}")

            plt.show()
            print("Plot generation complete.")

except ValueError as ve:
    print(f"\n--- ERROR: Input Data Issue ---")
    print(f"Details: {ve}")
except NameError as ne:
    print(f"\n--- ERROR: Configuration Issue ---")
    print(f"Details: {ne}")
except Exception as e:
    print(f"\n--- ERROR during plotting ---")
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()
