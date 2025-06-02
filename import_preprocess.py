
import numpy as np
import scipy.io
import mne
from sklearn.preprocessing import MinMaxScaler
import os
from tqdm import tqdm  # Importing tqdm for the progress bar
import numpy as np
import mne




def load_hs_mat(file_path, selected_channels):
    """
    Load EEG and kinematics data from a .mat file.
    """
    data = scipy.io.loadmat(file_path)
    hs = data["hs"][0, 0]
    eeg_data = hs["eeg"][0, 0]["sig"]  # EEG signals (samples, channels)
    channel_names = [ch[0] for ch in hs["eeg"][0, 0]["names"][0]]
    kin_columns = [20, 21, 22, 24, 25, 26, 28, 29, 30]
    kin_data = hs["kin"][0, 0]["sig"][:, kin_columns]
    
    # Select only required EEG channels
    channel_indices = [i for i, ch in enumerate(channel_names) if ch in selected_channels]
    eeg_data = eeg_data[:, channel_indices]
    
    return eeg_data, kin_data, selected_channels


def load_all_data(folder_path, selected_channels):
    """
    Load all EEG and kinematic data from dataset folder.
    """
    eeg_all, kin_all = [], []
    for p in range(1, 13):
        for s in range(1, 10):
            file_name = f"HS_P{p}_S{s}.mat"
            file_path = os.path.join(folder_path, file_name)
            if os.path.exists(file_path):
                eeg_data, kin_data, _ = load_hs_mat(file_path, selected_channels)
                eeg_all.append(eeg_data)
                kin_all.append(kin_data)
                print(f"Loaded: {file_name}")
            else:
                print(f"Missing: {file_name}")
    return eeg_all, kin_all


selected_channels = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
    "TP9", "CP5", "CP1", "CP2", "CP6", "TP10", "P7", "P3", "Pz", "P4", "P8",
    "PO9", "O1", "Oz", "O2", "PO10"]


# Load data
folder_path = "./"  # Update path

eeg_data, kin_data = load_all_data(folder_path, selected_channels)
# Save EEG data and Kinematics data to .npy files
eeg_data_path = "eeg_data.npy"
kin_data_path = "kin_data.npy"

# Convert to NumPy arrays and save
np.save(eeg_data_path, np.array(eeg_data))
np.save(kin_data_path, np.array(kin_data))

print(f"Saved EEG data to {eeg_data_path}")
print(f"Saved Kinematics data to {kin_data_path}")



eeg_data = np.load("eeg_data.npy", allow_pickle=True)
session_lengths = [session.shape[0] for session in eeg_data]
split_indices = np.cumsum(session_lengths[:-1])-1  # Compute split points




num_participants = 12
num_sessions = 108
num_channels = 32  # From the shape of individual elements

# Define 32 EEG channels
selected_channels = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
                     "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
                     "TP9", "CP5", "CP1", "CP2", "CP6", "TP10", "P7", "P3", "Pz",
                     "P4", "P8", "PO9", "O1", "Oz", "O2", "PO10"]

final_channels = ['F3', 'Fz', 'F4', 'FC5', 'FC1', 'FC2', 'FC6',
                  'C3', 'Cz', 'C4', 'CP5', 'CP1', 'CP2', 'CP6',
                  'P7', 'P3', 'Pz', 'P4', 'O1', 'Oz', 'O2']

fs = 500  # Original sampling rate




# Suppress MNE logging for cleaner output

# Create MNE Info object
info = mne.create_info(ch_names=selected_channels, sfreq=fs, ch_types="eeg")
raw_full = mne.io.RawArray(np.vstack(eeg_data).T, info)

ch_pos = np.array([
    [-2.7, 8.6, 3.6], [2.7, 8.6, 3.6], [-6.7, 5.2, 3.6], [-4.7, 6.2, 8], [0, 6.7, 9.5], [4.7, 6.2, 8], [6.7, 5.2, 3.6],
    [-5.5, 3.2, 6.6], [-3, 3.3, 11], [3, 3.3, 11], [5.5, 3.2, 6.6], [-7.8, 0, 3.6], [-6.1, 0, 9.7], [0, 0, 12], [6.1, 0, 9.7], [7.8, 0, 3.6],
    [-7.3, -2.5, 0], [-7.2, -2.7, 6.6], [-3, -3.2, 11], [3, -3.2, 11], [7.2, -2.7, 6.6], [7.3, -2.5, 0], [-6.7, -5.2, 3.6], [-4.7, -6.2, 8],
    [0, -6.7, 9.5], [4.7, -6.2, 8], [6.7, -5.2, 3.6], [-4.7, -6.7, 0], [-2.7, -8.6, 3.6], [0, -9, 3.6], [2.7, -8.6, 3.6], [4.7, -6.7, 0]
]) / 1000  # Convert from mm to meters


montage = mne.channels.make_dig_montage(ch_pos=dict(zip(selected_channels, ch_pos)), coord_frame="head")
raw_full.set_montage(montage)

raw_full.filter(l_freq=0.1, h_freq=40, fir_design="firwin")
raw_full.set_eeg_reference(ref_channels="average", projection=False)

raw_filename="eeg_raw.fif"
raw_full.save(raw_filename, overwrite=True)


mne.set_log_level('ERROR')


#   Run ICA Per Session

def run_ica_per_session(raw_session, session_index):
    """Runs ICA on a single session and removes artifacts."""
    print(f"Running ICA on session {session_index+1}...")

    # Get initial sample count
    n_samples_before = raw_session.n_times
    print(f"Samples before ICA: {n_samples_before}")

    # Run ICA
    ica = mne.preprocessing.ICA(n_components=32, random_state=97, max_iter=1000, method='fastica')
    ica.fit(raw_session)
    
    # Identify & Remove Artifacts
    eog_indices, _ = ica.find_bads_eog(raw_session, ch_name=["Fp1", "Fp2"])
    ica.exclude = eog_indices

    # Apply ICA Correction
    raw_ica = ica.apply(raw_session)

    # Get final sample count
    n_samples_after = raw_ica.n_times
    print(f"Samples after ICA: {n_samples_after}")

    # Check if the sample count changed
    if n_samples_before != n_samples_after:
        print(f"⚠️ WARNING: Sample count changed for session {session_index+1}!")

    return raw_ica

# ------------------------------- #
#  Step 3: Process All Sessions
# ------------------------------- #

def process_eeg_sessions(raw_filename, cleaned_filename, split_indices):
    """Loads raw EEG, runs ICA per session, and saves cleaned EEG as a single FIF file."""

    #  Load raw EEG from .fif file
    raw = mne.io.read_raw_fif(raw_filename, preload=True)

    #  Prepare session split points
    starts = [0] + list(split_indices)  
    ends = list(split_indices) + [raw.n_times]  
    print(starts)
    print(ends)

    #  Start ICA Cleaning Process
    print(" Starting ICA cleaning process...")
    cleaned_data = []

    for i, (start, end) in tqdm(enumerate(zip(starts, ends)), total=len(starts)):
        print(f"\n Processing session {i+1}/{len(starts)}... (Samples {start} to {end})")

        if start >= end:
            print(f" Skipping session {i+1} (Invalid range: start={start}, end={end})")
            continue  

        #  Select session data
        raw_session = raw.copy().crop(tmin=start / raw.info["sfreq"], tmax=end / raw.info["sfreq"])

        #  Run ICA
        raw_ica = run_ica_per_session(raw_session, i)

        #  Append cleaned data
        cleaned_data.append(raw_ica)

    #  Merge all cleaned sessions into a single MNE object
    raw_cleaned = mne.concatenate_raws(cleaned_data)

    #  Save the final cleaned EEG as a single `.fif` file
    print(f" Saving cleaned EEG to {cleaned_filename}...")
    raw_cleaned.save(cleaned_filename, overwrite=True)
    print(f" ICA cleaning process completed successfully! Cleaned data saved as {cleaned_filename}")

# ------------------------------- #
#  Step 4: Run Everything
# ------------------------------- #

#  Define filenames
raw_filename = "eeg_raw.fif"
cleaned_filename = "cleaned_eeg.fif"

#  Process EEG sessions and save cleaned data as `.fif`
process_eeg_sessions(raw_filename, cleaned_filename, split_indices)


final_channels = ['F3', 'Fz', 'F4', 'FC5', 'FC1', 'FC2', 'FC6',
                  'C3', 'Cz', 'C4', 'CP5', 'CP1', 'CP2', 'CP6',
                  'P7', 'P3', 'Pz', 'P4', 'O1', 'Oz', 'O2']

raw_cleaned = mne.io.read_raw_fif("cleaned_eeg.fif", preload=True)
# raw_cleaned.filter(l_freq=0.5, h_freq=3, picks="eeg", method="fir", fir_design="firwin")
raw_cleaned.pick_channels(final_channels)
raw_cleaned.save("filtered_clean_eeg_0.53.fif", overwrite=True)