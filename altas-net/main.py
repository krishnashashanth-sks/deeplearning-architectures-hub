import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import IrisDataset
from train import train_model
from evaluate import evaluate_model

# 1. Model import
from model import AltasModel
# 2. Data Preparation
# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')

# Scale features
scaler = StandardScaler()
X_scaled_array = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns)

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Create DataLoader instances
batch_size = 16
train_dataset = IrisDataset(X_train, y_train)
val_dataset = IrisDataset(X_val, y_val)
test_dataset = IrisDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# 3. Loss Function and Metrics
num_classes = y.nunique()
input_dim = X_train.shape[1]
hidden_dim = 64
output_dim = num_classes

model = AltasModel(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()

# 4. Optimizer and Scheduler
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
step_size = 10
gamma = 0.5
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 5. Training Loop
num_epochs = 50
train_losses,val_losses,train_accuracies,val_accuracies =train_model(num_epochs,model,train_loader,val_loader,optimizer,criterion,scheduler,device)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy over Epochs')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
print("Plots of training and validation metrics displayed.")

# 6. Evaluation and Inference
model.load_state_dict(torch.load('model_best_accuracy.pth'))
model.to(device)

model.eval()
test_running_loss,test_running_corrects,all_true_labels,all_predicted_labels=test_total_samples=evaluate_model(test_loader,model,criterion,device)

test_loss = test_running_loss / test_total_samples
test_accuracy = test_running_corrects / test_total_samples

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

all_true_labels_np = np.array(all_true_labels)
all_predicted_labels_np = np.array(all_predicted_labels)

print("\nClassification Report:")
print(classification_report(all_true_labels_np, all_predicted_labels_np, target_names=iris.target_names))

conf_matrix = confusion_matrix(all_true_labels_np, all_predicted_labels_np)
print("\nConfusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()