# Multi-modal-Signal-Integration-for-Enhanced-Sleep-Stage-Classification_new

Running the Sleep Stage Classification Algorithm
Paper: "Multi-modal Signal Integration for Enhanced Sleep Stage Classification: Leveraging EOG and 2-channel EEG Data with Advanced Feature Extraction"
Authors: Mahdi Samaee, Mehran Yazdi, Daniel Massicotte

This code is designed for step-by-step sequential execution by the user. In future versions, we aim to provide a more user-friendly and optimized version, focusing on computational cost and parallelization.

To run the code, follow these steps:

Prepare Data:
Copy the preprocessed data files of each subject into one folder seperately.

Wavelet Feature Extraction:
Execute wavelet_feature_extraction.py for all modalities of data.

STFT Feature Extraction:
Execute STFT_feature_K1.py to STFT_feature_K4.py for all modalities of data.

Generate Folds:
Run fold_generation.py to create 20 folders containing text files with training and validation data subject names.

Extract and Concatenate Fold Data (Wavelet and RAW):
Execute extractFoldData_concatenate.py for all wavelet extracted features and raw data.
Specify the number of folds and indicate whether you are working on validation data or training data.
This code should be run for all raw and wavelet data.

Extract and Concatenate Fold Data (STFT):
Execute extractFoldDataSTFT_concatenate.py for all STFT extracted features.
Specify the number of folds and indicate whether you are working on validation data or training data.
This code should be run for all raw and STFT data.

CNN Feature Extraction (RAW Data):
Run CNN_1D_RAW_K1.py to CNN_1D_RAW_K4.py for feature extraction from raw data.
Determine the data source names and output file names based on the working data.

CNN Feature Extraction (Wavelet Data):
Run CNN_1D_wavelet_K1.py to CNN_1D_wavelet_K4.py for feature extraction from wavelet data.
Determine the data source names and output file names based on the working data.

CNN Feature Extraction (STFT Data):
Run CNN_2D_STFT_K1.py to CNN_2D_STFT_K4.py for feature extraction from STFT data.
Determine the data source names and output file names based on the working data.

Final Steps:
After manually running all CNN-based feature extraction codes, you should have 36 training files and 36 validation files.
Depending on the desired classifier, run the relevant code. For instance, to use XGBoost for classification, execute classification_XGboost.py.
