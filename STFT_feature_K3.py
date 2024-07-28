import numpy as np
from scipy.signal import stft, hamming
import matplotlib.pyplot as plt
import os

# Defining the paths to the source and destination directories
source_dir = "data_ROC_A1"
destination_dir = "ROC_A1_STFT_features_K3"

# checking existance the destination directory 
os.makedirs(destination_dir, exist_ok=True)

# Defining the window size and overlap
window_size = int(4 * 100)  # 4 seconds of signal at 100 samples per second
overlap = int(window_size * 0.70)  # 70% overlap

# Defining the Hamming window
hamming_window = hamming(window_size)

# Iterating over all .npy files in the source directory
for filename in os.listdir(source_dir):
    if filename.endswith(".npy") and not filename.endswith("_label.npy"):
        # Loading the EEG data
        data = np.load(os.path.join(source_dir, filename))

        # Initializing an array to store STFT features
        stft_matrix = np.zeros((data.shape[0], 201, 26), dtype=np.complex64)

        # Computing the STFT and log-power spectrum for each EEG signal
        for i in range(data.shape[0]):
            # Computing the STFT using the defined parameters
            f, t, Zxx = stft(data[i], fs=100, window=hamming_window, nperseg=window_size, noverlap=overlap)

            # Computing the power spectrum from the STFT
            power_spectrum = np.abs(Zxx) ** 2

            # Computing the log-power spectrum
            log_power_spectrum = 10 * np.log10(power_spectrum)

            # Storing the log-power spectrum in the matrix
            stft_matrix[i, :, :] = log_power_spectrum

        # Constructing the destination filename
        base_name = filename.replace('.npz', '').replace('.npy', '')
        dest_filename = f"{base_name}.npy"

        # Saving the STFT features to the destination directory
        np.save(os.path.join(destination_dir, dest_filename), stft_matrix)

print("STFT feature extraction and saving completed.")
