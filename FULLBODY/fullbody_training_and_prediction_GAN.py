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



        
    import time
    import os
    from models import create_model

    class SimpleOptions:
        def __init__(self):
            # ---  Model Parameters ---
            self.model = 'pix2pix'
            self.input_nc = 60
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
            self.no_dropout = True      # Use dropout in the generator or not

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
            self.verbose = False

    # %%


    # --- 1. Get Training Options ---
    opt = SimpleOptions()

    # In your main training notebook cell
    # ... after opt = SimpleOptions()
    print("-" * 20)
    print(f"DEBUG: About to create model with num_downs = {opt.num_downs}")
    print("-" * 20)

    # --- 2. Create Save Directory ---
    save_directory = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(save_directory, exist_ok=True)

    # --- 3. Ensure DataLoaders are defined ---
    print(f'The number of training samples = {len(train_loader.dataset)}')
    print(f'The number of validation samples = {len(val_loader.dataset)}')

    # --- 4. Initialize Model ---
    model = create_model(opt)
    model.setup(opt)
    print(f"Model [{type(model).__name__}] was created successfully.")

    # --- Lists to store per-epoch loss values for plotting ---
    loss_history = {
        "G_GAN": [], "G_L1": [], "D_real": [], "D_fake": [], "val_G_L1": []
    }

    # --- 5. Main Training and Validation Loop ---
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

            
    # --- 6. PLOTTING ---
    # (Plotting code remains the same)
    # --- 6. PLOTTING (NEW) ---
    # This block will run after all training epochs are complete
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

    folder_name=f"0.1-40hz_gan_results_step_{STEP_SIZE}"
    os.makedirs(folder_name, exist_ok=True)

    file_name = f"participant_{TARGET_SUBJECT_KEY}_shift_{STEP_SIZE}loss_plots.png"
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

    if 'test_loader' in locals() and test_loader is not None:
        with torch.no_grad():
            for test_data in test_loader:
                eeg_data, kin_data = test_data
                
                # This part is specific to your GAN model's input structure
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


    
    with open(f'GAN_0.1-40hz.csv', 'a') as fd:
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

        file_name = f"participant_{TARGET_SUBJECT_KEY}_gan_continuous_plot.png"
        save_path = os.path.join(folder_name, file_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as {file_name}")

else:
    print("No test data was generated, skipping plot.")