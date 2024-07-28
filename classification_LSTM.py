from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import seaborn as sns
import matplotlib.patches as patches
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


scaler_minmax = MinMaxScaler(feature_range=(-1, 1))

####            loading data
######################### STFT features  #############################################################
# Train features
EEG_Fpz_Cz_SFFT_train_features = np.load('EEG_C3_A2_STFT_trainFeatures_K1_CNN2D.npy')
EEG_Fpz_Cz_SFFT_kernel2_train_features = np.load('EEG_C3_A2_STFT_trainFeatures_K2_CNN2D.npy')
EEG_Fpz_Cz_SFFT_kernel3_train_features = np.load('EEG_C3_A2_STFT_trainFeatures_K3_CNN2D.npy')
EEG_Fpz_Cz_SFFT_kernel4_train_features = np.load('EEG_C3_A2_STFT_trainFeatures_K4_CNN2D.npy')

EEG_Pz_Oz_STFT_train_features = np.load('EEG_F3_A2_STFT_trainFeatures_K1_CNN2D.npy')
EEG_Pz_Oz_STFT_kernel2_train_features = np.load('EEG_F3_A2_STFT_trainFeatures_K2_CNN2D.npy')
EEG_Pz_Oz_STFT_kernel3_train_features = np.load('EEG_F3_A2_STFT_trainFeatures_K3_CNN2D.npy')
EEG_Pz_Oz_STFT_kernel4_train_features = np.load('EEG_F3_A2_STFT_trainFeatures_K4_CNN2D.npy')

EOG_STFT_train_features = np.load('ROC_A1_STFT_trainFeatures_K1_CNN2D.npy')
EOG_STFT_kernel2_train_features = np.load('ROC_A1_STFT_trainFeatures_K2_CNN2D.npy')
EOG_STFT_kernel3_train_features = np.load('ROC_A1_STFT_trainFeatures_K3_CNN2D.npy')
EOG_STFT_kernel4_train_features = np.load('ROC_A1_STFT_trainFeatures_K4_CNN2D.npy')

# Test features
EEG_Fpz_Cz_SFFT_test_features = np.load('EEG_C3_A2_STFT_testFeatures_K1_CNN2D.npy')
EEG_Fpz_Cz_SFFT_kernel2_test_features = np.load('EEG_C3_A2_STFT_testFeatures_K2_CNN2D.npy')
EEG_Fpz_Cz_SFFT_kernel3_test_features = np.load('EEG_C3_A2_STFT_testFeatures_K3_CNN2D.npy')
EEG_Fpz_Cz_SFFT_kernel4_test_features = np.load('EEG_C3_A2_STFT_testFeatures_K4_CNN2D.npy')

EEG_Pz_Oz_STFT_test_features = np.load('EEG_F3_A2_STFT_testFeatures_K1_CNN2D.npy')
EEG_Pz_Oz_STFT_kernel2_test_features = np.load('EEG_F3_A2_STFT_testFeatures_K2_CNN2D.npy')
EEG_Pz_Oz_STFT_kernel3_test_features = np.load('EEG_F3_A2_STFT_testFeatures_K3_CNN2D.npy')
EEG_Pz_Oz_STFT_kernel4_test_features = np.load('EEG_F3_A2_STFT_testFeatures_K4_CNN2D.npy')

EOG_STFT_test_features = np.load('ROC_A1_STFT_testFeatures_K1_CNN2D.npy')
EOG_STFT_test_kernel2_features = np.load('ROC_A1_STFT_testFeatures_K2_CNN2D.npy')
EOG_STFT_test_kernel3_features = np.load('ROC_A1_STFT_testFeatures_K3_CNN2D.npy')
EOG_STFT_test_kernel4_features = np.load('ROC_A1_STFT_testFeatures_K4_CNN2D.npy')

################# wavelet features  #####################################

EEG_Fpz_Cz_wavelet_train_features = np.load('EEG_C3_A2_wavelet_trainFeatures_K1.npy')
EEG_Fpz_Cz_wavelet_kernel2_train_features = np.load('EEG_C3_A2_wavelet_trainFeatures_K2.npy')
EEG_Fpz_Cz_wavelet_kernel3_train_features = np.load('EEG_C3_A2_wavelet_trainFeatures_K3.npy')
#EEG_Fpz_Cz_wavelet_kernel4_train_features = np.load('EEG_C3_A2_wavelet_trainFeatures_K4.npy')


EEG_Pz_Oz_wavelet_train_features = np.load('EEG_F3_A2_wavelet_trainFeatures_K1.npy')
EEG_Pz_Oz_wavelet_kernel2_train_features = np.load('EEG_F3_A2_wavelet_trainFeatures_K2.npy')
EEG_Pz_Oz_wavelet_kernel3_train_features = np.load('EEG_F3_A2_wavelet_trainFeatures_K3.npy')
#EEG_Pz_Oz_wavelet_kernel4_train_features = np.load('EEG_F3_A2_wavelet_trainFeatures_K4.npy')

EOG_wavelet_train_features = np.load('ROC_A1_wavelet_trainFeatures_K1.npy')
EOG_wavelet_kernel2_train_features = np.load('All_features/ROC_A1_wavelet_trainFeatures_K2.npy')
EOG_wavelet_kernel3_train_features = np.load('ROC_A1_wavelet_trainFeatures_K3.npy')
#EOG_wavelet_kernel4_train_features = np.load('ROC_A1_wavelet_trainFeatures_K4.npy')

### Test features
EEG_Fpz_Cz_wavelet_test_features = np.load('EEG_C3_A2_wavelet_testFeatures_K1.npy')
EEG_Fpz_Cz_wavelet_kernel2_test_features = np.load('EEG_C3_A2_wavelet_testFeatures_K2.npy')
EEG_Fpz_Cz_wavelet_kernel3_test_features = np.load('EEG_C3_A2_wavelet_testFeatures_K3.npy')
#EEG_Fpz_Cz_wavelet_kernel4_test_features = np.load('EEG_C3_A2_wavelet_testFeatures_K4.npy')

EEG_Pz_Oz_wavelet_test_features = np.load('EEG_F3_A2_wavelet_testFeatures_K1.npy')
EEG_Pz_Oz_wavelet_kernel2_test_features = np.load('EEG_F3_A2_wavelet_testFeatures_K2.npy')
EEG_Pz_Oz_wavelet_kernel3_test_features = np.load('EEG_F3_A2_wavelet_testFeatures_K3.npy')
#EEG_Pz_Oz_wavelet_kernel4_test_features = np.load('EEG_F3_A2_wavelet_testFeatures_K4.npy')

EOG_wavelet_test_features = np.load('ROC_A1_wavelet_testFeatures_K1.npy')
EOG_wavelet_kernel2_test_features = np.load('ROC_A1_wavelet_testFeatures_K2.npy')
EOG_wavelet_kernel3_test_features = np.load('ROC_A1_wavelet_testFeatures_K3.npy')
#EOG_wavelet_kernel4_test_features = np.load('ROC_A1_wavelet_testFeatures_K4.npy')

############################# RAW features  ###############################################

EEG_Fpz_Cz_RAW_train_features = np.load('EEG_C3_A2_RAW_trainFeatures_K1.npy')
EEG_Fpz_Cz_RAW_kernel2_train_features = np.load('EEG_C3_A2_RAW_trainFeatures_K2.npy')
EEG_Fpz_Cz_RAW_kernel3_train_features = np.load('EEG_C3_A2_RAW_trainFeatures_K3.npy')
EEG_Fpz_Cz_RAW_kernel4_train_features = np.load('EEG_C3_A2_RAW_trainFeatures_K4.npy')

EEG_Fpz_Cz_RAW_test_features = np.load('EEG_C3_A2_RAW_testFeatures_K1.npy')
EEG_Fpz_Cz_RAW_kernel2_test_features = np.load('EEG_C3_A2_RAW_testFeatures_K2.npy')
EEG_Fpz_Cz_RAW_kernel3_test_features = np.load('EEG_C3_A2_RAW_testFeatures_K3.npy')
EEG_Fpz_Cz_RAW_kernel4_test_features = np.load('EEG_C3_A2_RAW_testFeatures_K4.npy')

EEG_Pz_Oz_RAW_train_features = np.load('EEG_F3_A2_RAW_trainFeatures_K1.npy')
EEG_Pz_Oz_RAW_kernel2_train_features = np.load('EEG_F3_A2_RAW_trainFeatures_K2.npy')
EEG_Pz_Oz_RAW_kernel3_train_features = np.load('EEG_F3_A2_RAW_trainFeatures_K3.npy')
EEG_Pz_Oz_RAW_kernel4_train_features = np.load('EEG_F3_A2_RAW_trainFeatures_K4.npy')


EEG_Pz_Oz_RAW_test_features = np.load('EEG_F3_A2_RAW_testFeatures_K1.npy')
EEG_Pz_Oz_RAW_kernel2_test_features = np.load('EEG_F3_A2_RAW_testFeatures_K2.npy')
EEG_Pz_Oz_RAW_kernel3_test_features = np.load('EEG_F3_A2_RAW_testFeatures_K3.npy')
EEG_Pz_Oz_RAW_kernel4_test_features = np.load('EEG_F3_A2_RAW_testFeatures_K4.npy')

EOG_RAW_train_features = np.load('ROC_A1_RAW_trainFeatures_K1.npy')
EOG_RAW_kernel2_train_features = np.load('ROC_A1_RAW_trainFeatures_K2.npy')
EOG_RAW_kernel3_train_features = np.load('ROC_A1_RAW_trainFeatures_K3.npy')
EOG_RAW_kernel4_train_features = np.load('ROC_A1_RAW_trainFeatures_K4.npy')

EOG_RAW_test_features = np.load('ROC_A1_RAW_testFeatures_K1.npy')
EOG_RAW_kernel2_test_features = np.load('ROC_A1_RAW_testFeatures_K2.npy')
EOG_RAW_kernel3_test_features = np.load('ROC_A1_RAW_testFeatures_K3.npy')
EOG_RAW_kernel4_test_features = np.load('ROC_A1_RAW_testFeatures_K4.npy')


###########################  labels  #######################################

train_labels = np.load('concatenated_train_labels.npy')
test_labels = np.load('concatenated_val_labels.npy')
###################################################################
#################################
class_names = ["Wake", "N1", "N2", "N3", "REM"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


####################################################################################################
###########################             concatenating features

train_features_final1 = np.hstack(
    (EEG_Fpz_Cz_SFFT_train_features, EEG_Pz_Oz_STFT_train_features, EOG_STFT_train_features,
     EEG_Fpz_Cz_SFFT_kernel2_train_features, EEG_Pz_Oz_STFT_kernel2_train_features, EOG_STFT_kernel2_train_features,
     EEG_Fpz_Cz_SFFT_kernel3_train_features, EEG_Pz_Oz_STFT_kernel3_train_features, EOG_STFT_kernel3_train_features,
     EEG_Fpz_Cz_SFFT_kernel4_train_features, EEG_Pz_Oz_STFT_kernel4_train_features, EOG_STFT_kernel4_train_features))

test_features_final1 = np.hstack((EEG_Fpz_Cz_SFFT_test_features, EEG_Pz_Oz_STFT_test_features, EOG_STFT_test_features,
                                  EEG_Fpz_Cz_SFFT_kernel2_test_features, EEG_Pz_Oz_STFT_kernel2_test_features,
                                  EOG_STFT_test_kernel2_features,
                                  EEG_Fpz_Cz_SFFT_kernel3_test_features, EEG_Pz_Oz_STFT_kernel3_test_features,
                                  EOG_STFT_test_kernel3_features,
                                  EEG_Fpz_Cz_SFFT_kernel4_test_features, EEG_Pz_Oz_STFT_kernel4_test_features,
                                  EOG_STFT_test_kernel4_features))

#####################

train_features_final2 = np.hstack(
    (EEG_Fpz_Cz_wavelet_train_features, EEG_Pz_Oz_wavelet_train_features, EOG_wavelet_train_features,
     EEG_Fpz_Cz_wavelet_kernel2_train_features, EEG_Pz_Oz_wavelet_kernel2_train_features,
     EOG_wavelet_kernel2_train_features,
     EEG_Fpz_Cz_wavelet_kernel3_train_features, EEG_Pz_Oz_wavelet_kernel3_train_features,
     EOG_wavelet_kernel3_train_features))

test_features_final2 = np.hstack(
    (EEG_Fpz_Cz_wavelet_test_features, EEG_Pz_Oz_wavelet_test_features, EOG_wavelet_test_features,
     EEG_Fpz_Cz_wavelet_kernel2_test_features, EEG_Pz_Oz_wavelet_kernel2_test_features,
     EOG_wavelet_kernel2_test_features,
     EEG_Fpz_Cz_wavelet_kernel3_test_features, EEG_Pz_Oz_wavelet_kernel3_test_features,
     EOG_wavelet_kernel3_test_features))

###############################



train_features_final3 = np.hstack((EEG_Fpz_Cz_RAW_train_features, EEG_Pz_Oz_RAW_train_features, EOG_RAW_train_features,
                                   EEG_Fpz_Cz_RAW_kernel2_train_features, EEG_Pz_Oz_RAW_kernel2_train_features,
                                   EOG_RAW_kernel2_train_features,
                                   EEG_Fpz_Cz_RAW_kernel3_train_features, EEG_Pz_Oz_RAW_kernel3_train_features,
                                   EOG_RAW_kernel3_train_features,
                                   EEG_Fpz_Cz_RAW_kernel4_train_features, EEG_Pz_Oz_RAW_kernel4_train_features,
                                   EOG_RAW_kernel4_train_features))

test_features_final3 = np.hstack((EEG_Fpz_Cz_RAW_test_features, EEG_Pz_Oz_RAW_test_features, EOG_RAW_test_features,
                                  EEG_Fpz_Cz_RAW_kernel2_test_features, EEG_Pz_Oz_RAW_kernel2_test_features,
                                  EOG_RAW_kernel2_test_features,
                                  EEG_Fpz_Cz_RAW_kernel3_test_features, EEG_Pz_Oz_RAW_kernel3_test_features,
                                  EOG_RAW_kernel3_test_features,
                                  EEG_Fpz_Cz_RAW_kernel4_test_features, EEG_Pz_Oz_RAW_kernel4_test_features,
                                  EOG_RAW_kernel4_test_features))

#######################################################################

################        Dimension Reduction with PCA
n_components = 50
pca = PCA(n_components=n_components)


train_features_final1 = pca.fit_transform(train_features_final1)
test_features_final1 = pca.transform(test_features_final1)

######

n_components = 50
pca = PCA(n_components=n_components)


train_features_final2 = pca.fit_transform(train_features_final2)
test_features_final2 = pca.transform(test_features_final2)
#######################

n_components = 50
pca = PCA(n_components=n_components)


train_features_final3 = pca.fit_transform(train_features_final3)
test_features_final3 = pca.transform(test_features_final3)

####################  concatenating all dimensionality reduced features   #################################
train_features_final = np.hstack((train_features_final1, train_features_final2, train_features_final3))

test_features_final = np.hstack((test_features_final1, test_features_final2, test_features_final3))


########################
scaler = StandardScaler()
train_features_final = scaler.fit_transform(train_features_final.reshape(-1, train_features_final.shape[-1])).reshape(train_features_final.shape)
test_features_final = scaler.transform(test_features_final.reshape(-1, test_features_final.shape[-1])).reshape(test_features_final.shape)




#####################    LSTM  ###############################################
train_labels_final = train_labels
test_labels_final = test_labels


# Normalizing the data
scaler = StandardScaler()
train_features_final = scaler.fit_transform(train_features_final.reshape(-1, train_features_final.shape[-1])).reshape(train_features_final.shape)
test_features_final = scaler.transform(test_features_final.reshape(-1, test_features_final.shape[-1])).reshape(test_features_final.shape)

# Defining the sequence length (number of segments per sequence)
sequence_length = 8

# Creating overlapping sequences of EEG segments
def create_overlapping_sequences(data, labels, sequence_length):
    sequences = []
    sequence_labels = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])
        sequence_labels.append(labels[i + sequence_length - 1])
    return np.array(sequences), np.array(sequence_labels)

X_train_seq, y_train_seq = create_overlapping_sequences(train_features_final, train_labels_final, sequence_length)
X_test_seq, y_test_seq = create_overlapping_sequences(test_features_final, test_labels_final, sequence_length)


# Defining a custom Dataset for EEG data
class EEGDataset(Dataset):
    def __init__(self, eeg_data, labels):
        self.eeg_data = eeg_data
        self.labels = labels

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        sample = {'data': self.eeg_data[idx], 'label': self.labels[idx]}
        return sample

# Defining the LSTM model with Batch Normalization and Dropout
class EEGNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EEGNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.bn(out[:, -1, :])
        out = self.dropout(out)
        return out

# Defining the final classification MLP with Dropout
class MLPClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(num_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Hyperparameters
input_size = X_train_seq.shape[2]
hidden_size = 128
num_layers = 3
num_classes = 5
num_epochs = 30
batch_size = 32
#learning_rate = 0.001
#weight_decay = 1e-4

# Convert to PyTorch tensors
train_data = torch.tensor(X_train_seq, dtype=torch.float32)
train_labels = torch.tensor(y_train_seq, dtype=torch.long)
test_data = torch.tensor(X_test_seq, dtype=torch.float32)
test_labels = torch.tensor(y_test_seq, dtype=torch.long)

# Create dataset and DataLoader
train_dataset = EEGDataset(train_data, train_labels)
test_dataset = EEGDataset(test_data, test_labels)

# Split train dataset into training and validation sets
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# Initialize models, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eeg_net = EEGNet(input_size, hidden_size, num_layers).to(device)
mlp_classifier = MLPClassifier(hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(list(eeg_net.parameters()) + list(mlp_classifier.parameters()))

# Initialize lists to store losses and accuracy for plotting
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training and validation loop
for epoch in range(num_epochs):
    eeg_net.train()
    mlp_classifier.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(train_loader, total=len(train_loader))

    for sample in pbar:
        inputs = sample['data'].to(device)
        labels = sample['label'].to(device)
        optimizer.zero_grad()

        features = eeg_net(inputs)
        outputs = mlp_classifier(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}")
        pbar.set_postfix(loss=running_loss / len(train_loader), accuracy=100 * correct / total)

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    train_accuracy = 100 * correct / total
    train_accuracies.append(train_accuracy)
    print(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

    # Validation
    eeg_net.eval()
    mlp_classifier.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for sample in val_loader:
            inputs = sample['data'].to(device)
            labels = sample['label'].to(device)

            features = eeg_net(inputs)
            outputs = mlp_classifier(features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    val_accuracy = 100 * correct / total
    val_accuracies.append(val_accuracy)
    print(f"Epoch {epoch + 1}/{num_epochs} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

print("Training Finished")




# Evaluation and Confusion Matrix Plotting Function
def evaluate_and_plot_confusion_matrix(eeg_net, mlp_classifier, test_loader, class_names, save_path="ConfusionMatrix_LSTM.png"):
    eeg_net.eval()
    mlp_classifier.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for sample in test_loader:
            inputs = sample['data'].to(device)
            labels = sample['label'].to(device)
            features = eeg_net(inputs)
            outputs = mlp_classifier(features)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    precisions, recalls, f1_scores, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
    MF1 = np.mean(f1_scores)

    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test Precision: {precision:.4f}')
    print(f'Test Recall: {recall:.4f}')
    print(f'Test F1 Score: {f1:.4f}')
    print("Precision for each class:", precisions)
    print("Recall for each class:", recalls)
    print("F1-score for each class:", f1_scores)
    print("Macro F1-score (MF1):", MF1)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix using heatmap
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix_LSTM")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")

    # Calculate accuracy for each class
    class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

    # Calculate overall accuracy
    overall_accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)

    # Add accuracy for each class to the heatmap
    for i, accuracy in enumerate(class_accuracies):
        plt.text(i + 0.5, i + 0.1, f' {accuracy:.2%}', ha='center', va='center', color='red')
        box = patches.Rectangle((i + 0.25, i), 0.55, 0.2, linewidth=1, edgecolor='white', facecolor='white')
        plt.gca().add_patch(box)

    # Add overall accuracy to the plot
    plt.text(0, -0.12, f'Overall Accuracy: {overall_accuracy:.2%}', size=10, ha="center", transform=plt.gca().transAxes, color='red')
    plt.text(1, -0.12, f'Macro F1-score: {MF1:.2%}', size=10, ha="center", transform=plt.gca().transAxes, color='red')

    # Save and show the plot
    plt.savefig("E:/MyResearchOnSleep/MyCode/1/sleep-cassette_NEW/ISRUC-SLEEP_Dataset_2/fold_1/All_features/RESULTS/ConfusionMatrix_AllFeatures_LSTM.png")
    plt.show()

# Example usage
class_names = ["Wake", "N1", "N2", "N3", "REM"]
evaluate_and_plot_confusion_matrix(eeg_net, mlp_classifier, test_loader, class_names)

