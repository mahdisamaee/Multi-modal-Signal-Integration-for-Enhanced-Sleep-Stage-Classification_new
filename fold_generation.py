import os
import numpy as np

# Create 20 folders for cross-validation
base_dir = "ISRUC-SLEEP_Dataset_2"
num_folds = 20
num_subjects = 126
subjects = [f"subject_{i}" for i in range(1, num_subjects + 1)]
np.random.shuffle(subjects)  # Shuffle subjects to ensure randomness

# Create directories and text files for each fold
for fold in range(1, num_folds + 1):
    fold_dir = os.path.join(base_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    test_subjects = subjects[(fold - 1) * (num_subjects // num_folds): fold * (num_subjects // num_folds)]
    train_subjects = [subject for subject in subjects if subject not in test_subjects]

    train_file_path = os.path.join(fold_dir, "train_files.txt")
    test_file_path = os.path.join(fold_dir, "val_files.txt")

    with open(train_file_path, 'w') as train_file:
        for subject in train_subjects:
            train_file.write(subject + "\n")

    with open(test_file_path, 'w') as test_file:
        for subject in test_subjects:
            test_file.write(subject + "\n")
