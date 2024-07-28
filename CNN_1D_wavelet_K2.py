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
train_data_main = np.load("concatenated_train_data_wavelet_EEG_C3_A2.npy")

Test_data_main = np.load("concatenated_val_data_wavelet_EEG_C3_A2.npy")
Test_label_main = np.load("concatenated_val_labels.npy")

################
# Convert complex matrix to float64 by taking the real part
train_data_main = train_data_main.real.astype(np.float64)

Test_data_main = Test_data_main.real.astype(np.float64)

# Find the maximum and minimum values in the float matrix
max_value = train_data_main.max()
min_value = train_data_main.min()

# Scale the float matrix to the range [-1, 1]
train_data_main = 2 * (train_data_main - min_value) / (max_value - min_value) - 1
########################

Test_data_main = 2 * (Test_data_main - min_value) / (max_value - min_value) - 1
#######################
train_data_EEG, test_data_EEG, train_labels_EEG, test_labels_EEG = train_test_split(train_data_main, train_labels_main, test_size=0.05, random_state=42, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reshaping the input data
train_data_EEG = torch.Tensor(train_data_EEG).unsqueeze(1)  # Adding a channel dimension
test_data_EEG = torch.Tensor(test_data_EEG).unsqueeze(1)

train_data_main = torch.Tensor(train_data_main).unsqueeze(1)  # Adding a channel dimension
Test_data_main = torch.Tensor(Test_data_main).unsqueeze(1)

# Preprocessing the labels
num_classes = 5
train_labels_EEG = torch.LongTensor(train_labels_EEG)
test_labels_EEG = torch.LongTensor(test_labels_EEG)


train_labels_main = torch.LongTensor(train_labels_main)
Test_label_main = torch.LongTensor(Test_label_main)
# Creating TensorDatasets for training and testing
train_dataset = TensorDataset(train_data_EEG, train_labels_EEG)
test_dataset = TensorDataset(test_data_EEG, test_labels_EEG)


train_dataset_main = TensorDataset(train_data_main, train_labels_main)
test_dataset_main = TensorDataset(Test_data_main, Test_label_main)
# Creating DataLoader for training and testing
batch_size = 32
train_loader_EEG = DataLoader(train_dataset, batch_size=batch_size)
test_loader_EEG = DataLoader(test_dataset, batch_size=batch_size)


train_loader_main = DataLoader(train_dataset_main, batch_size=batch_size)
test_loader_main = DataLoader(test_dataset_main, batch_size=batch_size)
# Creating the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv1d(1, 64, kernel_size=100, stride=12, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=8, stride=2)
        self.drop = nn.Dropout(p=0.5)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=1)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=1)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.flatten = nn.Flatten()
        self.fc1= nn.Linear(3712,128)

    def forward(self, x):
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.pool1(x)
        x = self.drop(x)

        x = self.conv2(x)
        x = F.gelu(x)

        x = self.conv3(x)
        x = F.gelu(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x= self.fc1(x)
        return x


# Define the final classification MLP
class MLPClassifier2(nn.Module):
    def __init__(self, num_classes):
        super(MLPClassifier2, self).__init__()
        self.fc2 = nn.Linear(50,50)
        self.fc3 = nn.Linear(50,50)
        self.fc4 = nn.Linear(128,num_classes)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        # = self.fc2(x)
        #x = self.fc3(x)
        x = self.relu1(x)
        x = self.fc4(x)
        return x




model2 = Net()
classifier2 = MLPClassifier2(num_classes=5)  

# Defining loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(model2.parameters()) + list(classifier2.parameters()))

# Training the model
epochs = 30
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model2.to(device)
classifier2.to(device)

for epoch in range(epochs):
    model2.train()
    classifier2.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(train_loader_EEG, total=len(train_loader_EEG))
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        features  = model2(inputs)
        outputs = classifier2(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_description(f"Epoch {epoch + 1}/{epochs}")
        pbar.set_postfix(loss=running_loss / len(train_loader_EEG), accuracy=100 * correct / total)

    accuracy = 100 * correct / total
    print(
        f"Epoch {epoch + 1}/{epochs} - Training Loss: {running_loss / len(train_loader_EEG):.4f}, Training Accuracy: {accuracy:.2f}%")

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
        for inputs, labels in test_loader_EEG:
            inputs, labels = inputs.to(device), labels.to(device)

            features = model2(inputs)
            outputs = classifier2(features)


            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())

        val_loss /= len(test_loader_EEG)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}/{epochs} - Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

print("training Finished")
print("Extracting train and test features based on trained network")


model2.eval()

features_EEG_all= []
with torch.no_grad():
    for inputs, labels in train_loader_main:
        inputs, labels = inputs.to(device), labels.to(device)

        features_EEG = model2(inputs)
        features_EEG_all.append(features_EEG.cpu())
# Concatenate the arrays vertically using numpy.vstack()
features_EEG_final = np.vstack(features_EEG_all)
######
features_test_EEG_all= []
with torch.no_grad():
    for inputs, labels in test_loader_main:
        inputs, labels = inputs.to(device), labels.to(device)

        features_EEG = model2(inputs)
        features_test_EEG_all.append(features_EEG.cpu())

# Concatenate the arrays vertically using numpy.vstack()
features_test_EEG_final = np.vstack(features_test_EEG_all)




#saving model
print("saving model ...")
# Save the trained model and classifier
torch.save(model2.state_dict(), "model2_CNN_1D_wavelet_EEG_C3_A2_K2.pth")
torch.save(classifier2.state_dict(), "classifier2_CNN_1D_wavelet_EEG_C3_A2_K2.pth")



print("saving features ...")
np.save("EEG_C3_A2_wavelet_trainFeatures_K2.npy", features_EEG_final)
np.save("EEG_C3_A2_wavelet_testFeatures_K2.npy",features_test_EEG_final)












