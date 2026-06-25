import matplotlib.pyplot as plt
from model import FourCastNet
from dataset import SpatioTemporalDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import torch
import torch.nn as nn
from train import train
from inference import inference_function

in_channels = 2
out_channels = 2
modes1 = 12
modes2 = 12
width = 20

# Instantiate the FourCastNet model
# Let's say we want 2 FNO layers per block
fourcastnet_model = FourCastNet(in_channels, out_channels, modes1, modes2, width, num_fno_blocks=2)

# --- 1. Generate Dummy Spatio-Temporal Data ---
# Let's simulate some time-series 2D data, e.g., atmospheric pressure and temperature maps
# Data format: (num_samples, num_time_steps, channels, height, width)

num_samples = 100 # Number of independent sequences/simulations
num_total_time_steps = 50 # Total time steps in each sequence
in_channels = 2 # e.g., pressure, temperature
height = 64
width_dim = 64

# Generate random data (replace with your actual data loading)
dummy_full_data = torch.randn(
    num_samples, num_total_time_steps, in_channels, height, width_dim
)

# Create dataset instances
train_ratio = 0.8
split_idx = int(num_samples * train_ratio)

train_data = dummy_full_data[:split_idx]
val_data = dummy_full_data[split_idx:]

train_dataset = SpatioTemporalDataset(train_data, input_sequence_length, output_sequence_length)
val_dataset = SpatioTemporalDataset(val_data, input_sequence_length, output_sequence_length)

# --- 3. Create PyTorch DataLoaders ---
batch_size =256#Using the batch_size defined in your previous example

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# --- 4. Define Hyperparameters and Training Setup ---

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Loss function (Mean Squared Error for regression tasks)
criterion = nn.MSELoss()

# Optimizer (Adam is a common choice for deep learning models)
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Number of training epochs
num_epochs = 5

train_losses,val_losses=train(num_epochs,model,train_dataloader,optimizer,criterion,device)


plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

print("\n--- Model Inference ---")
# Assuming 'model', 'val_loader', and 'device' are already defined from previous cells
inputs_val, targets_val, predictions_val = inference_function(model, val_loader, device)

print(f"Shape of validation inputs: {inputs_val.shape}")
print(f"Shape of validation targets: {targets_val.shape}")
print(f"Shape of validation predictions: {predictions_val.shape}")