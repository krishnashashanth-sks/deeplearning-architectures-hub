# Neccessary  libraries for implementation of PointNet
#pip install torch_geometric trimesh
from torch_geometric.loader import DataLoader
import os
from dataset import train_dataset,test_dataset
from train import train
from test import test
import torch
from model import PointNet2Classification

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create data loaders
batch_size = 32 # You can adjust this based on your GPU memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, optimizer, and loss function
num_classes = train_dataset.num_classes
model = PointNet2Classification(num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.NLLLoss() # Negative Log Likelihood Loss for classification

# Training loop
num_epochs = 50 # You can adjust the number of epochs
print("Starting training...")
for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train(model,train_loader,optimizer,criterion,device)
    test_loss, test_acc = test(model,test_loader,criterion,device)
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
print("Training complete.")