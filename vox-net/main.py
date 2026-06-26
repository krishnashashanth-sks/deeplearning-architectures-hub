import torch.optim as optim
import pandas as pd
import numpy as np
from model import VoxNet
from train import *
import torch
from torch.utils.data import DataLoader
from dataset import train_dataset,val_dataset,test_dataset,class_to_idx
import torch.nn as nn

# Define batch size for DataLoaders
batch_size = 32

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Instantiate the VoxNet model (assuming VoxNet class is defined)
# We pass num_classes from the loaded data
model = VoxNet(num_classes=len(class_to_idx)).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training parameters
num_epochs = 20
best_val_accuracy = 0.0

print("Starting full training loop...")

for epoch in range(num_epochs):
    train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)

    print(f"Epoch {epoch+1}/{num_epochs}:\n"\
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}\n"\
          f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_voxnet_model.pth')
        print

voxel_dim = 32 # Assuming a 32x32x32 voxel grid
num_classes = 10 # Based on ModelNet10

# Generate new dummy input data for prediction (e.g., 5 samples)
num_prediction_samples = 5
new_dummy_input = torch.randn(num_prediction_samples, 1, voxel_dim, voxel_dim, voxel_dim)

print(f"New dummy input shape for prediction: {new_dummy_input.shape}")

# Load the best trained model
# Ensure the model architecture is the same as when it was saved
model_for_prediction = VoxNet(num_classes=num_classes).to(device) # Reuse the 'device' from training
model_for_prediction.load_state_dict(torch.load('best_voxnet_model.pth'))
model_for_prediction.eval() # Set the model to evaluation mode

print("Best trained VoxNet model loaded for prediction.")

# Move input to the same device as the model
new_dummy_input = new_dummy_input.to(device);

# Perform inference
with torch.no_grad(): # No need to calculate gradients during inference
    predictions = model_for_prediction(new_dummy_input)

# Get predicted class labels
predicted_labels = torch.argmax(predictions, dim=1)

print(f"Model predictions (logits):\n{predictions}")
print(f"Predicted class labels:\n{predicted_labels}")