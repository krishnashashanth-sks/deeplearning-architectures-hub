import matplotlib.pyplot as plt
import torch.optim as optim
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from model import GAT
from test import test
from train import train
import torch.nn as nn
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils import to_networkx

# Load the Cora dataset
# Apply a normalization transform to node features for better performance
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0] # Get the first graph in the dataset

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Prepare data by moving it to the selected device
data = data.to(device)

hidden_channels = 8 # Example value for hidden dimension per head
num_heads = 8 # Example number of attention heads
dropout = 0.6 # Example dropout rate

model = GAT(
    in_channels=dataset.num_features,
    hidden_channels=hidden_channels,
    out_channels=dataset.num_classes,
    num_heads=num_heads,
    dropout=dropout
).to(device)

print(f'GAT model instantiated successfully with {dataset.num_features} input features, ')
print(f'  {hidden_channels} hidden channels (per head), {dataset.num_classes} output classes, ')
print(f'  {num_heads} attention heads, and {dropout} dropout rate.')
# Define loss function
criterion = nn.CrossEntropyLoss() # Suitable for multi-class classification

# Define optimizer
learning_rate = 0.005 # Common learning rate for GNNs
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4) # Adam optimizer with weight decay

epochs = 100 # Set the number of epochs for training

# Lists to store metrics for plotting
train_losses = []
train_accuracies = []
val_accuracies = []
test_accuracies = []

print("Starting training...")
for epoch in range(1, epochs + 1):
    loss = train(model,optimizer,data,criterion) # Call the train function
    train_acc, val_acc, test_acc = test(model,data) # Call the test function

    # Store metrics
    train_losses.append(loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    test_accuracies.append(test_acc)

    # Log results for the first epoch and every tenth epoch thereafter
    if epoch == 1 or epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

print("Training finished!")
# Plotting the training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
plt.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy')
plt.title('Training and Validation Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the training loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='red')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# --- Create a sample graph for prediction ---
# This is a very small graph for demonstration purposes.
# It needs to have node features (x) and edge_index.

# Number of nodes in our sample graph
sample_num_nodes = 5
# Number of features per node should match the training data
sample_in_channels = dataset.num_features

# Generate random features for the sample nodes
# Shape: [sample_num_nodes, sample_in_channels]
sample_x = torch.randn(sample_num_nodes, sample_in_channels).to(device)

# Define some edges for the sample graph
# Shape: [2, num_sample_edges]
# Example: node 0 connected to 1, 1 to 2, 2 to 3, 3 to 4, 4 to 0
sample_edge_index = torch.tensor([
    [0, 1, 2, 3, 4],
    [1, 2, 3, 4, 0]
], dtype=torch.long).to(device)

# Create a PyG Data object for the sample graph
sample_data = Data(x=sample_x, edge_index=sample_edge_index)

print("Sample graph created:")
print(sample_data)

# --- Perform prediction using the trained GAT model ---
model.eval() # Set the model to evaluation mode
with torch.no_grad(): # Disable gradient calculation for inference
    predictions = model(sample_data.x, sample_data.edge_index)

# The output 'predictions' will be raw logits. Apply softmax to get probabilities
probabilities = torch.softmax(predictions, dim=1)

# Get the predicted class for each node
predicted_classes = probabilities.argmax(dim=1)

print("\nPredictions for the sample graph nodes:")
for i in range(sample_num_nodes):
    print(f"Node {i}: Predicted Class = {predicted_classes[i].item()}, Probabilities = {probabilities[i].cpu().numpy()}")

# Convert PyG Data object to NetworkX graph
g = to_networkx(sample_data, to_undirected=True)

# Get predicted classes for coloring nodes
node_colors = predicted_classes.cpu().numpy()

# Map class labels to distinct colors
num_classes = dataset.num_classes
cmap = plt.colormaps['viridis'] # Use the recommended way to get colormap

# Create a figure and an axes object
fig, ax = plt.subplots(figsize=(8, 8))

pos = nx.spring_layout(g, seed=42) # For consistent layout

# Draw nodes and edges
nodes_drawn = nx.draw_networkx_nodes(g, pos, node_color=node_colors, cmap=cmap,
                                      node_size=700, alpha=0.8, ax=ax) # Store the PathCollection object
nx.draw_networkx_edges(g, pos, alpha=0.5, ax=ax)
nx.draw_networkx_labels(g, pos, font_size=10, font_color='black', ax=ax)

ax.set_title('Sample Graph Nodes Colored by Predicted Class')
ax.axis('off')

# Now, use the mappable returned by nx.draw_networkx_nodes (nodes_drawn) for the colorbar
# The mappable should be the PathCollection object returned by nx.draw_networkx_nodes
cbar = fig.colorbar(nodes_drawn, ticks=range(num_classes), ax=ax, label='Predicted Class', shrink=0.7)
cbar.ax.set_yticklabels(range(num_classes)) # Set explicit tick labels for clarity

plt.show()