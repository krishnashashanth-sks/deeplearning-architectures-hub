import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from model import LTSTANet
from train import train_lista_net
import torch.optim as optim
from inference import predict_sparse_code
from torch.utils.data import TensorDataset,DataLoader
import torch.nn.functional as F

# 1. Define the parameters for the synthetic sparse coding dataset
signal_dim = 64  # m: Dimension of the input signal y
code_dim = 128   # n: Dimension of the sparse code x (overcomplete dictionary)
num_samples = 10000 # Number of synthetic data samples
sparsity_level = 5 # Number of non-zero elements in each sparse code
noise_std = 0.1  # Standard deviation of the additive noise
batch_size = 64

D = torch.randn(signal_dim, code_dim)
D = F.normalize(D, p=2, dim=0) # Normalize columns to unit L2 norm

X_true = torch.zeros(num_samples, code_dim)
for i in range(num_samples):
    # Randomly select sparsity_level indices
    non_zero_indices = torch.randperm(code_dim)[:sparsity_level]
    # Assign random values to these selected indices
    X_true[i, non_zero_indices] = torch.randn(sparsity_level)

# Generate noisy signals Y = X_true @ D.T + noise
Y = torch.matmul(X_true, D.T) + noise_std * torch.randn(num_samples, signal_dim)

# Create TensorDataset and DataLoader
dataset = TensorDataset(Y, X_true)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"Dataset parameters defined: signal_dim={signal_dim}, code_dim={code_dim}, num_samples={num_samples}, sparsity_level={sparsity_level}, noise_std={noise_std}, batch_size={batch_size}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model parameters
n_layers = 15 # Number of unrolled layers

# Instantiate the LISTANet model
model = LTSTANet(signal_dim, code_dim, n_layers).to(device)
print(f"LISTANet model instantiated with {n_layers} layers.")

# Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training parameters
num_epochs = 50

print("Starting training...")
train_lista_net(model, optimizer, loss_fn, data_loader, num_epochs, device)
print("Training finished.")
# Select a few example signals to visualize
num_examples_to_plot = 5

# Get the example signals (y) and their true sparse codes (x_true)
example_y = Y[:num_examples_to_plot]
example_x_true = X_true[:num_examples_to_plot]

# Predict sparse codes for these example signals
example_x_pred = predict_sparse_code(model, example_y, device)

# Plotting
plt.figure(figsize=(15, 3 * num_examples_to_plot))

for i in range(num_examples_to_plot):
    plt.subplot(num_examples_to_plot, 1, i + 1)
    plt.stem(example_x_true[i].cpu().numpy(), linefmt='b-', markerfmt='bo', basefmt=' ',
             label='True Sparse Code')
    plt.stem(example_x_pred[i].cpu().numpy(), linefmt='r--', markerfmt='rx', basefmt=' ',
             label='Predicted Sparse Code')
    plt.title(f'Sample {i+1}: True vs. Predicted Sparse Code')
    plt.xlabel('Coefficient Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()