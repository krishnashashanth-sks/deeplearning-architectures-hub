import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pennylane import numpy as np # Keep pennylane's numpy for quantum operations
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from train import train_model
from evaluate import evaluate_model
from visualize import plot_decision_boundary
from model import HybridQuantumNeuralNetwork

# Set a random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42) # Set PyTorch seed

# Generate a synthetic dataset (make_moons)
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

# Scale the features to be suitable for quantum encoding (e.g., -pi to pi)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = X_scaled * np.pi / 2 # Scale to approximately -pi/2 to pi/2, suitable for rotation gates

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Visualize the dataset
plt.scatter(X_scaled[y == 0, 0], X_scaled[y == 0, 1], color='red', marker='o', label='Class 0')
plt.scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1], color='blue', marker='x', label='Class 1')
plt.title('Synthetic Moon Dataset (Scaled)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
# Instantiate the hybrid model
model = HybridQuantumNeuralNetwork(X_train)
# Convert numpy arrays to PyTorch tensors for DataLoader
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Define DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 1. Define the loss function for binary classification
loss_fn = nn.BCELoss()

# 2. Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


num_epochs = 50

train_losses = []
val_losses = []
val_accuracies = []

print("Starting training...")
for epoch in range(num_epochs):
    # Train for one epoch
    current_train_loss = train_model(model, train_loader, loss_fn, optimizer, device)
    train_losses.append(current_train_loss)

    # Evaluate on the test set
    current_val_loss, current_val_accuracy = evaluate_model(model, test_loader, loss_fn, device)
    val_losses.append(current_val_loss)
    val_accuracies.append(current_val_accuracy)

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {current_train_loss:.4f}, Val Loss: {current_val_loss:.4f}, Val Acc: {current_val_accuracy:.4f}")

print("Training complete!")

# Evaluate the model on the test set
test_loss, test_accuracy = evaluate_model(model, test_loader, loss_fn, device)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Model Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Model Accuracy History')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# Visualize the decision boundary on the test set
plot_decision_boundary(model, X_test_tensor.cpu().numpy(), y_test_tensor, device, title='QuanNN Decision Boundary (Test Set)')