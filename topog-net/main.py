import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from utils import extract_topological_features
from model import TopoGNN
from train import train_model
from evaluate import evaluate_model

# Define a transform to add a 3D position attribute to the data
# This is often useful for molecular graphs, though not strictly required for basic TDA
transform = T.Compose([
    T.Distance(norm=False),
    T.Cartesian(),
])

# Load the QM9 dataset
# root specifies the directory where the dataset will be stored
# pre_transform and transform are used to process the raw data
dataset = QM9(root='./data/QM9', pre_transform=transform)

print(f"Dataset: {dataset}")
print(f"Number of graphs: {len(dataset)}")
print(f"Number of features: {dataset.num_features}")
print(f"Number of classes: {dataset.num_classes}")

# --- Apply to the dataset ---
# We will process a small subset first to demonstrate and avoid long computation times
# For the full dataset, this would be applied to all graphs.

processed_dataset = []
subset_size = 100 # Process only the first 100 graphs for demonstration
print(f"Extracting topological features for the first {subset_size} graphs...")

# Ensure dataset is loaded, assuming 'dataset' variable from previous cell is available
# If running this cell independently, ensure QM9 dataset is loaded first.

for i, data_graph in enumerate(dataset[:subset_size]):
    if i % 10 == 0:
        print(f"Processing graph {i+1}/{subset_size}")
    topo_feat = extract_topological_features(data_graph)
    data_graph.x_topo = topo_feat # Attach as a new attribute
    processed_dataset.append(data_graph)

# Manual split of the processed_dataset (which is a list of Data objects)
num_graphs = len(processed_dataset)
num_train = int(num_graphs * 0.7)
num_val = int(num_graphs * 0.15)
num_test = num_graphs - num_train - num_val

# Ensure reproducibility of the split
# We'll shuffle the dataset before splitting to ensure a random distribution
# This requires converting to a list if not already to use slicing after shuffling
import random
random.seed(42) # For reproducibility
random.shuffle(processed_dataset)

train_dataset = processed_dataset[:num_train]
val_dataset = processed_dataset[num_train:num_train + num_val]
test_dataset = processed_dataset[num_train + num_val:]

print(f"\nDataset Split:")
print(f"  Training graphs: {len(train_dataset)}")
print(f"  Validation graphs: {len(val_dataset)}")
print(f"  Test graphs: {len(test_dataset)}")

# Verify the first graph in the training set still has the x_topo attribute
if len(train_dataset) > 0:
    print(f"First graph in training set (x_topo shape): {train_dataset[0].x_topo.shape}")

# 1. Initialize the TopoGNN model and move it to the appropriate device (CPU or GPU).
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model parameters (obtained from previous execution)
num_node_features = 11
num_topo_features = 800
hidden_channels = 128
num_classes = 19

model = TopoGNN(num_node_features, num_topo_features, hidden_channels, num_classes).to(device)
print(f"\nModel initialized and moved to {device}:")
print(model)

# 2. Define the loss function (MSELoss) and instantiate an optimizer (e.g., Adam) with a suitable learning rate.
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(f"\nLoss function: {loss_fn}")
print(f"Optimizer: {optimizer}")

# 3. Create PyTorch Geometric DataLoader objects for the datasets.
batch_size = 32 # Define batch size

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, follow_batch=['x_topo'])
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, follow_batch=['x_topo'])
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, follow_batch=['x_topo'])

print(f"\nDataLoaders created with batch size {batch_size}:")
print(f"  Train DataLoader: {len(train_loader)} batches")
print(f"  Validation DataLoader: {len(val_loader)} batches")
print(f"  Test DataLoader: {len(test_loader)} batches")

epochs = 50 # Number of training epochs
best_val_loss = float('inf')

print("\nStarting training...")
for epoch in range(1, epochs + 1):
    train_loss = train_model(model,train_loader,optimizer,loss_fn,device)
    val_loss = evaluate_model(model,val_loader,loss_fn,device)

    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Optional: Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        print(f'  New best validation loss. Model saved.')

print("\nTraining complete.")

model.eval()
predictions = []
ground_truths = []

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.x_topo, data.batch)
        predictions.append(out.cpu().numpy())
        ground_truths.append(data.y.squeeze(1).cpu().numpy())

predictions = np.concatenate(predictions, axis=0)
ground_truths = np.concatenate(ground_truths, axis=0)

# For simplicity, let's visualize the first target property (index 0)
target_idx = 0

plt.figure(figsize=(10, 6))
plt.scatter(ground_truths[:, target_idx], predictions[:, target_idx], alpha=0.6)
plt.xlabel(f'Ground Truth (Target {target_idx})')
plt.ylabel(f'Predictions (Target {target_idx})')
plt.title(f'Predictions vs. Ground Truth for Target {target_idx} on Test Set')
plt.plot([min(ground_truths[:, target_idx]), max(ground_truths[:, target_idx])], 
         [min(ground_truths[:, target_idx]), max(ground_truths[:, target_idx])], 
         color='red', linestyle='--', label='Ideal Prediction')
plt.legend()
plt.grid(True)
plt.show()