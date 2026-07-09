import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
import torch
from evaluate import evaluate_model
from inference import predict_model
from model import DCRNN
from train import train_model
from utils import compute_scaled_diffusion_matrix

### Data
# 1. Define the graph structure (adjacency matrix)
num_nodes = 7 # A small number of nodes

# Create a random graph using networkx
G = nx.gnp_random_graph(num_nodes, p=0.4, seed=42) # n=num_nodes, p=probability of edge creation

# Ensure the graph is connected (optional, but good for demonstrating diffusion)
# If it's not connected, add edges to make it connected
if not nx.is_connected(G):
    print("Graph is not connected. Adding edges to connect it.")
    # Find connected components and add edges between them
    components = list(nx.connected_components(G))
    for i in range(len(components) - 1):
        node1 = list(components[i])[0]
        node2 = list(components[i+1])[0]
        G.add_edge(node1, node2)

# Get the adjacency matrix from the networkx graph
adj_matrix = nx.to_numpy_array(G)

print(f"Number of nodes: {num_nodes}")
print("Adjacency Matrix (numpy):")
print(adj_matrix)

# 2. Convert adjacency matrix to PyTorch Geometric's edge_index format
# Get row and column indices of non-zero elements (edges)
row_indices, col_indices = np.where(adj_matrix > 0)

# Stack them to form edge_index
# PyG's edge_index is typically (2, num_edges)
edge_index = torch.tensor([row_indices, col_indices], dtype=torch.long)

print(f"Edge Index (torch.LongTensor, shape {edge_index.shape}):")
print(edge_index)
print(f"Number of nodes (derived from adj_matrix shape): {num_nodes}")
# 3. Generate dummy spatio-temporal node features
num_time_steps = 100 # Total number of time steps
num_features = 2   # Number of features per node (e.g., traffic speed, volume)

# Create a tensor of shape (num_time_steps, num_nodes, num_features)
# using random values for simplicity. For real data, these would be actual observations.
node_features = torch.randn(num_time_steps, num_nodes, num_features, dtype=torch.float32)

print(f"Node Features (torch.Tensor, shape {node_features.shape}):")
print(node_features[:5, :2, :]) # Print first 5 time steps for first 2 nodes and all features
# 4. Split time-series data into input and target sequences
history_length = 12  # Number of past time steps to observe
predict_length = 3   # Number of future time steps to predict

# Ensure there are enough time steps for the split
if num_time_steps < history_length + predict_length:
    raise ValueError("Not enough time steps for the specified history and prediction lengths.")

input_sequences = []
target_sequences = []

for i in range(num_time_steps - history_length - predict_length + 1):
    # Input sequence: (history_length, num_nodes, num_features)
    input_seq = node_features[i : i + history_length]
    input_sequences.append(input_seq)

    # Target sequence: (predict_length, num_nodes, num_features)
    target_seq = node_features[i + history_length : i + history_length + predict_length]
    target_sequences.append(target_seq)

input_sequences = torch.stack(input_sequences)
target_sequences = torch.stack(target_sequences)

print(f"Input Sequences (torch.Tensor, shape {input_sequences.shape}):")
print(input_sequences[0, :2, :2, :]) # Print first 2 time steps, first 2 nodes, all features of the first input sequence
print(f"\nTarget Sequences (torch.Tensor, shape {target_sequences.shape}):")
print(target_sequences[0, :2, :2, :]) # Print first 2 time steps, first 2 nodes, all features of the first target sequence

diffusion_matrix_f, diffusion_matrix_b = compute_scaled_diffusion_matrix(adj_matrix, k=2)

# Parameters for the DCRNN model
input_size = num_features # Number of features per node
hidden_size = 64          # Hidden state size for DCRNNCell

# Instantiate the DCRNN model
dcrnn_model = DCRNN(
    input_size=input_size,
    hidden_size=hidden_size,
    num_nodes=num_nodes,
    history_length=history_length,
    predict_length=predict_length,
    K=2 # Assuming K=2 from previous DiffusionConv examples
)

# Ensure diffusion matrices are on the correct device if using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dcrnn_model.to(device)
input_sequences_on_device = input_sequences.to(device)
diffusion_matrix_f_on_device = diffusion_matrix_f.to(device)
diffusion_matrix_b_on_device = diffusion_matrix_b.to(device)

criterion = nn.MSELoss()
learning_rate = 0.001
optimizer = optim.Adam(dcrnn_model.parameters(), lr=learning_rate)

print(f"Loss function (criterion): {criterion}")
print(f"Optimizer: {optimizer}")

### Data Splitting ,evaluating and training
# 1. Determine sizes for splitting
num_samples = input_sequences.shape[0]
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

num_train = int(num_samples * train_ratio)
num_val = int(num_samples * val_ratio)
num_test = num_samples - num_train - num_val # Ensure all samples are used

# 2. Generate random indices for shuffling
indices = torch.randperm(num_samples)

# 3. Split indices into train, validation, and test sets
train_indices = indices[:num_train]
val_indices = indices[num_train : num_train + num_val]
test_indices = indices[num_train + num_val :]

# 4. Create training, validation, and test data subsets
X_train, y_train = input_sequences[train_indices], target_sequences[train_indices]
X_val, y_val = input_sequences[val_indices], target_sequences[val_indices]
X_test, y_test = input_sequences[test_indices], target_sequences[test_indices]

# 5. Create TensorDataset objects
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

# 6. Create DataLoader objects
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")

#7.Training
num_epochs = 50

train_model(num_epochs,dcrnn_model,train_loader,optimizer,diffusion_matrix_f_on_device,diffusion_matrix_b_on_device,criterion,device)

#8.Evaluation
test_loss = evaluate_model(
    dcrnn_model,
    test_loader,
    criterion,
    diffusion_matrix_f_on_device,
    diffusion_matrix_b_on_device
)

print(f"Final Test Loss (MSE): {test_loss:.4f}")

#9.Inference
test_predictions = predict_model(
    dcrnn_model,
    test_loader,
    diffusion_matrix_f_on_device,
    diffusion_matrix_b_on_device
)

print(f"Shape of predictions on the test set: {test_predictions.shape}")