import matplotlib.pyplot as plt
import torch.optim as optim
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from evaluate import test
from inference import predict
from model import DGCNN
from train import train
import torch
import torch.nn as nn

# Define transformations
num_points = 1024 # Number of points per sample
pre_transform = T.NormalizeScale() # Normalize point cloud to fit into a unit sphere
transform = T.FixedPoints(num_points) # Sample a fixed number of points

# Instantiate the ModelNet40 dataset for training
print("Loading ModelNet40 training dataset...")
train_dataset = ModelNet(root='./data/ModelNet40', name='40', train=True, transform=transform, pre_transform=pre_transform)
print(f"ModelNet40 training dataset loaded with {len(train_dataset)} samples.")

# Instantiate the ModelNet40 dataset for testing
print("Loading ModelNet40 test dataset...")
test_dataset = ModelNet(root='./data/ModelNet40', name='40', train=False, transform=transform, pre_transform=pre_transform)
print(f"ModelNet40 test dataset loaded with {len(test_dataset)} samples.")

batch_size = 32 # Define your desired batch size

# Create DataLoader for training set
print("Creating training DataLoader...")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print(f"Training DataLoader created with batch size {batch_size}.")

# Create DataLoader for testing set
print("Creating testing DataLoader...")
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # No need to shuffle test data
print(f"Testing DataLoader created with batch size {batch_size}.")

#  Initialize the DGCNN model
# Assuming point clouds have 3 coordinates (x, y, z) as input features
in_channels = 3
# ModelNet40 has 40 classes
num_classes = 40
k_neighbors = 20 # k for KNNGraph
embed_dim = 1024 # Embedding dimension before classification head
drop_prob = 0.5 # Dropout probability

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DGCNN(in_channels=in_channels, num_classes=num_classes, k=k_neighbors, embed_dim=embed_dim, drop_prob=drop_prob).to(device)
print(f"Model initialized and moved to {device}.")

#  Define a loss function
criterion = nn.CrossEntropyLoss()

#  Choose an optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50 # You can adjust the number of epochs

# Lists to store metrics for visualization
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

print(f"Starting training for {num_epochs} epochs...")

for epoch in range(1, num_epochs + 1):
    print(f"\nEpoch {epoch}/{num_epochs}")
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, num_points, in_channels, device)
    test_loss, test_acc = test(model, test_loader, criterion,num_points, in_channels, device)

    # Store metrics
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    print(f"End of Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

print("Training complete!")

# Plotting Loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
plt.title('Training and Test Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plotting Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
plt.title('Training and Test Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Get a sample point cloud from the test dataset
sample_idx = 0
sample_data = test_dataset[sample_idx] # This returns a Data object from PyG

# The point cloud data is in sample_data.x, and the true label in sample_data.y
input_point_cloud = sample_data.x # Shape: (num_points, in_channels)
true_label = sample_data.y.item()

print(f"Input point cloud shape: {input_point_cloud.shape}")
print(f"True label: {true_label}")

# Perform inference
logits, predicted_class_idx = predict(model, input_point_cloud, device)

print(f"Predicted logits shape: {logits.shape}")
print(f"Predicted class index: {predicted_class_idx.item()}")

if hasattr(test_dataset, 'categories'):
    predicted_class_name = test_dataset.categories[predicted_class_idx.item()]
    true_class_name = test_dataset.categories[true_label]
    print(f"Predicted class name: {predicted_class_name}")
    print(f"True class name: {true_class_name}")
else:
    print("Class names not available in dataset object.")