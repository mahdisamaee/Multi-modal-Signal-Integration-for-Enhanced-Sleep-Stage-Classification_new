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





###################################################################################
####################################################################################
#####################    AdaBoost  #################################################
# Creating a base classifier (Decision Tree) for AdaBoost


# Defining the base classifier
base_classifier = DecisionTreeClassifier(max_depth=3)

# Creating an AdaBoost classifier with the base classifier and specified hyperparameters
adaboost_classifier = AdaBoostClassifier(base_estimator=base_classifier, n_estimators=130, learning_rate=0.1)

# Fit the AdaBoost classifier to the training data
adaboost_classifier.fit(train_features_final, train_labels)

# Predict classes on the test data using the fitted model
predictions = adaboost_classifier.predict(test_features_final)

# Calculating accuracy
accuracy = accuracy_score(test_labels, predictions)
print("Accuracy of optimized AdaBoost:", accuracy)

# Calculating confusion matrix
conf_matrix = confusion_matrix(test_labels, predictions)

precisions, recalls, f1_scores, _ = precision_recall_fscore_support(test_labels, predictions, average=None)
MF1 = np.mean(f1_scores)

# Print metrics
print("Precision for each class:", precisions)
print("Recall for each class:", recalls)
print("F1-score for each class:", f1_scores)
print("Macro F1-score (MF1):", MF1)

# Plot confusion matrix using heatmap
plt.figure(figsize=(8, 6))

###################################
#ax = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names)
# Calculate percentage matrix
percent_matrix = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
percent_str_matrix = np.array([['{:.2f}%'.format(value) for value in row] for row in percent_matrix])
ax = sns.heatmap(conf_matrix, annot=percent_str_matrix, fmt="", cmap="Greens", xticklabels=class_names, yticklabels=class_names)
######################################

plt.title("Confusion Matrix_AdaBoost")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")


# Calculating accuracy for each class
class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

# Calculating overall accuracy
overall_accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)

# Adding overall accuracy and MF1-score to the plot
plt.text(0, -0.12, f'Overall Accuracy: {overall_accuracy:.2%}', size=10, ha="center", transform=plt.gca().transAxes, color='red')
plt.text(1, -0.12, f'Macro F1-score: {MF1:.2%}', size=10, ha="center", transform=plt.gca().transAxes, color='red')

# Save and show the plot
plt.savefig("ConfusionMatrix_AllFeatures_AdaBoost.png")
plt.show()




