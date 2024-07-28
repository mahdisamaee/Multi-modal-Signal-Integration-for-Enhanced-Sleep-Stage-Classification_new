import os
import numpy as np

# Define the directories
fold_dir = "ISRUC-SLEEP_Dataset_2/fold_1"  
eog_data_dir = "ISRUC-SLEEP_Dataset_2/ROC_A1_STFT_features_K4"  

# Path to the train_files.txt
train_file_path = os.path.join(fold_dir, "train_files.txt")

# Initialize a list to hold the concatenated data
all_data = []

# Read the train_files.txt to get the list of training data filenames
with open(train_file_path, 'r') as file:
    train_files = file.readlines()

# Remove any potential newline characters from filenames
train_files = [filename.strip() + ".npy" for filename in train_files]

# Loop through each filename and load the corresponding .npy file
for filename in train_files:
    data_path = os.path.join(eog_data_dir, filename)

    if os.path.exists(data_path):
        # Load the .npy data file
        data = np.load(data_path)

        # Append the data to the all_data list
        all_data.append(data)
    else:
        print(f"File {data_path} not found. Skipping...")

# Concatenate all the data vertically
if all_data:
    concatenated_data = np.vstack(all_data)

    # Save the concatenated data if needed
    output_path = os.path.join(fold_dir, "concatenated_ROC_A1_train_data_STFT_K4.npy")
    np.save(output_path, concatenated_data)

    print(f"Concatenated data shape: {concatenated_data.shape}")
    print(f"Data saved to: {output_path}")
else:
    print("No data was loaded. Please check the file paths and filenames.")
