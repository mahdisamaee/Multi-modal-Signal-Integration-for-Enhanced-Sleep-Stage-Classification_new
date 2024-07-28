import os
import pywt
import numpy as np

# Defining the paths to the source and destination directories
source_dir = "data_ROC_A1"
destination_dir = "ROC_A1_wavelet_features"

# checking existance of destination directory 
os.makedirs(destination_dir, exist_ok=True)

# Defining wavelet parameters
wavelet = 'db38'  # selecting the wavelet function
level = 5  # selecting the decomposition level

# Iterating over all .npy files in the source directory
for filename in os.listdir(source_dir):
    if filename.endswith(".npy") and not filename.endswith("_label.npy"):
        # Loading the EEG data
        eeg_data = np.load(os.path.join(source_dir, filename))

        # Initializing an array to store wavelet features
        wavelet_features = []

        # Apply wavelet transform to each EEG signal row
        for signal in eeg_data:
            coeffs = pywt.wavedec(signal, wavelet, level=level)
            features = np.concatenate(coeffs)
            wavelet_features.append(features)

        # Converting the list of features into a NumPy array
        wavelet_features = np.array(wavelet_features)

        # Constructing the destination filename
        base_name = filename.replace('.npz', '').replace('.npy', '')
        dest_filename = f"{base_name}.npy"

        # Save the wavelet features to the destination directory
        np.save(os.path.join(destination_dir, dest_filename), wavelet_features)

print("Wavelet feature extraction and saving completed.")
