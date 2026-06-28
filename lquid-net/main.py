from model import LiquidNeuralNetwork
from torch.utils.data import TensorDataset,DataLoader,random_split
import torch
import torch.nn as nn
from train import train
device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
hidden_size = 64
input_size = 32
output_dim=10
sequence_length=5
num_samples = 100 # Total number of dummy data samples

# 2. Generate dummy initial hidden states (h0)
# Shape: (num_samples, hidden_size)
dummy_h0 = torch.randn(num_samples, hidden_size, device=device)
print(f"Shape of dummy_h0: {dummy_h0.shape}")

# 3. Generate dummy input sequences (x_sequence)
# Shape: (num_samples, sequence_length, input_size)
dummy_x_sequence = torch.randn(num_samples, sequence_length, input_size, device=device)
print(f"Shape of dummy_x_sequence: {dummy_x_sequence.shape}")

# 4. Generate dummy time points (t)
# Shape: (sequence_length,)
# Time points are typically the same for all samples in a batch for LNNs
dummy_t = torch.linspace(0., 1., sequence_length, device=device)
# Since t is the same for all samples, we'll store it once and expand if needed in DataLoader collate_fn
print(f"Shape of dummy_t: {dummy_t.shape}")

# 5. Generate dummy target output (target_outputs)
# Shape: (num_samples, output_dim)
dummy_target_outputs = torch.randn(num_samples, output_dim, device=device)
print(f"Shape of dummy_target_outputs: {dummy_target_outputs.shape}")

# 6. Combine these dummy tensors into a TensorDataset
# Note: dummy_t is a single tensor, not part of the per-sample data in TensorDataset
dummy_dataset = TensorDataset(dummy_h0, dummy_x_sequence, dummy_target_outputs)
print(f"Combined dummy data into TensorDataset with {len(dummy_dataset)} samples.")

# 7. Split the generated dummy dataset into training and validation sets
train_size = int(0.8 * num_samples)
val_size = num_samples - train_size

generator = torch.Generator().manual_seed(42) # For reproducibility
train_dataset, val_dataset = random_split(dummy_dataset, [train_size, val_size], generator=generator)

print(f"Dataset split: Training samples = {len(train_dataset)}, Validation samples = {len(val_dataset)}")

# Create DataLoaders (optional, but good for verification)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Created DataLoader for training with {len(train_dataloader)} batches.")
print(f"Created DataLoader for validation with {len(val_dataloader)} batches.")

lnn_model = LiquidNeuralNetwork(hidden_size, input_size, output_dim).to(device)

loss_function = nn.MSELoss() # Suitable for regression-like dummy_target_outputs
optimizer = torch.optim.Adam(lnn_model.parameters(), lr=0.001)

num_epochs = 10

train(num_epochs,lnn_model,train_dataloader,val_dataloader,dummy_t,loss_function,optimizer,device)