import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.transforms import ToTensor
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import torch.nn.functional as F

train_labels_main = np.load("concatenated_train_labels.npy")
train_data_main = np.load("concatenated_EEG_F3_A2_train_data_STFT_K2.npy")

test_data_main = np.load("concatenated_EEG_F3_A2_val_data_STFT_K2.npy")
test_labels_main = np.load("concatenated_val_labels.npy")
################
# Convert complex matrix to float64 by taking the real part
train_data_main = train_data_main.real.astype(np.float64)

# Find the maximum and minimum values in the float matrix
max_value = train_data_main.max()
min_value = train_data_main.min()

# Scale the float matrix to the range [-1, 1]
train_data_main = 2 * (train_data_main - min_value) / (max_value - min_value) - 1
########################
test_data_main = 2 * (test_data_main - min_value) / (max_value - min_value) - 1



train_data_EEG_Pz_Oz, test_data_EEG_Pz_Oz, train_labels_EEG_Pz_Oz, test_labels_EEG_Pz_Oz = train_test_split(train_data_main, train_labels_main, test_size=0.05, random_state=42, shuffle=True)

class ImageDataset2(Dataset):
    def __init__(self, data1, labels):
        self.data1 = data1
        self.labels = labels

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        image = self.data1[idx]
        label = self.labels[idx]
        return image, label


class CNNClassifier2(nn.Module):
    def __init__(self):
        super(CNNClassifier2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(19840, 128)
        #self.relu3 = nn.ReLU()
        #self.fc2 = nn.Linear(128, 5)  # 5 output classes

    def forward(self, x):
        x = x.float()  # Convert input to float data type
        x = x.unsqueeze(1)  # Add a single channel dimension
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc1(x)
        #x = self.relu3(x)
        #x = self.fc2(x)
        return x


# Define the final classification MLP
class MLPClassifier2(nn.Module):
    def __init__(self, num_classes):
        super(MLPClassifier2, self).__init__()
        self.relu1 = nn.ReLU()
        self.fc = nn.Linear(128, num_classes) 

    def forward(self, x):
        x = self.relu1(x)
        x = self.fc(x)
        return x






train_dataset = ImageDataset2(train_data_EEG_Pz_Oz, train_labels_EEG_Pz_Oz)
val_dataset = ImageDataset2(test_data_EEG_Pz_Oz, test_labels_EEG_Pz_Oz)

train_dataset_main = ImageDataset2(train_data_main, train_labels_main)
test_dataset_main = ImageDataset2(test_data_main, test_labels_main)




# Create data loaders for batch processing
batch_size = 32
train_loader_EEG_Pz_Oz = DataLoader(train_dataset, batch_size=batch_size)
test_loader_EEG_Pz_Oz = DataLoader(val_dataset, batch_size=batch_size)


train_loader_main = DataLoader(train_dataset_main, batch_size=batch_size)
test_loader_main = DataLoader(test_dataset_main, batch_size=batch_size)
# Create an instance of the CNN model
model2 = CNNClassifier2()
classifier2 = MLPClassifier2(num_classes=5) 

# Move the model to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model2.to(device)
classifier2.to(device)
# Define the loss function and optimizer
criterion2 = nn.CrossEntropyLoss()
optimizer2 = optim.Adam(list(model2.parameters()) + list(classifier2.parameters()))



from tqdm import tqdm

# Training loop
epochs = 30
for epoch in range(epochs):
    model2.train()
    classifier2.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader_EEG_Pz_Oz, total=len(train_loader_EEG_Pz_Oz))
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer2.zero_grad()
        features = model2(images)
        outputs = classifier2(features)
        loss = criterion2(outputs, labels.long())  # Convert labels to LongTensor
        loss.backward()
        optimizer2.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_description(f"Epoch {epoch + 1}/{epochs}")
        pbar.set_postfix(loss=running_loss / len(train_loader_EEG_Pz_Oz), accuracy=100 * correct / total)

    accuracy = 100 * correct / total
    print(
        f"Epoch {epoch + 1}/{epochs} - Training Loss: {running_loss / len(train_loader_EEG_Pz_Oz):.4f}, Training Accuracy: {accuracy:.2f}%")

    # Validation
    model2.eval()
    classifier2.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    # define lists to store the true labels and predicted labels
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for images, labels in test_loader_main:
            images, labels = images.to(device), labels.to(device)
            features = model2(images)
            outputs = classifier2(features)
            loss = criterion2(outputs, labels.long())  # Convert labels to LongTensor
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

    val_loss /= len(test_loader_main)
    accuracy = correct / total
    print(f"Epoch {epoch+1}/{epochs} - Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")


print("training Finished")
print("Extracting train and test features based on trained network")

# Validation
model2.eval()


features_EEG_Pz_Oz_all = []

with torch.no_grad():
    for images, labels in train_loader_main:
        images, labels = images.to(device), labels.to(device)
        features_EEG_Pz_Oz = model2(images)
        features_EEG_Pz_Oz_all.append(features_EEG_Pz_Oz.cpu())

# Concatenate the arrays vertically using numpy.vstack()
features_EEG_Pz_Oz_final = np.vstack(features_EEG_Pz_Oz_all)
###
features_test_EEG_Pz_Oz_all = []
with torch.no_grad():
    for images, labels in test_loader_main:
        images, labels = images.to(device), labels.to(device)
        features_EEG_Pz_Oz = model2(images)
        features_test_EEG_Pz_Oz_all.append(features_EEG_Pz_Oz.cpu())

# Concatenate the arrays vertically using numpy.vstack()
features_test_EEG_Pz_Oz_final = np.vstack(features_test_EEG_Pz_Oz_all)
##############################
#saving model
print("saving model ...")
# Save the trained model and classifier
# Save the trained model and classifier
torch.save(model2.state_dict(), "model2_CNN_2D_STFT_EEG_F3_A2_K2.pth")
torch.save(classifier2.state_dict(), "classifier2_CNN_2D_STFT_EEG_F3_A2_K2.pth")


print("saving features ...")
np.save("EEG_F3_A2_STFT_trainFeatures_K2_CNN2D.npy", features_EEG_Pz_Oz_final)
np.save("EEG_F3_A2_STFT_testFeatures_K2_CNN2D.npy",features_test_EEG_Pz_Oz_final)