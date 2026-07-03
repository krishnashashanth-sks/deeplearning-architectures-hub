import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from model import ConvNeXt
from train import *
from utils import *

device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
# Define model parameters
model_kwargs = {
    "in_chans": 3,
    "num_classes": 1000,
    "depths": [3, 3, 9, 3],
    "dims": [96, 192, 384, 768]
}

# Instantiate the ConvNeXt model
model = ConvNeXt(**model_kwargs).to(device)
# Load CIFAR-10 training dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
print(f"CIFAR-10 Training Dataset loaded: {len(train_dataset)} samples")

# Load CIFAR-10 test dataset
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True)
print(f"CIFAR-10 Test Dataset loaded: {len(test_dataset)} samples")
# Define transformations for training data (with augmentation)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Define transformations for testing data (without augmentation)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

print("Image transformations defined successfully.")
# Apply transformations to the datasets
train_dataset.transform = transform_train
test_dataset.transform = transform_test

# Create DataLoaders
batch_size = 64 # Using 64 as specified or 128 as an alternative

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print(f"Train DataLoader created with batch size {batch_size}, {len(train_loader)} batches.")
print(f"Test DataLoader created with batch size {batch_size}, {len(test_loader)} batches.")
# Define the loss function for multi-class classification
criterion = nn.CrossEntropyLoss()

print("Loss function (CrossEntropyLoss) defined successfully.")
# Define the optimizer
# AdamW is commonly used for modern CNN architectures like ConvNeXt
learning_rate = 4e-3 # A common starting learning rate for ConvNeXt
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print(f"Optimizer (AdamW) defined successfully with learning rate: {learning_rate}")
num_epochs = 1# Define the total number of epochs for training

# Initialize lists to store metrics
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

print("Starting training...")

for epoch in range(1, num_epochs + 1):
    print(f"\nEpoch {epoch}/{num_epochs}")

    # Train for one epoch
    epoch_train_loss, epoch_train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_accuracy)

    # Evaluate the model after each epoch
    epoch_val_loss, epoch_val_accuracy = evaluate_model(model, test_loader, criterion, device)
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_accuracy)

    # Print summary for the epoch
    print(f'Epoch {epoch} Summary: Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.4f}')

print("Training finished.")
# Set the model to evaluation mode
model.eval()

# Get a batch of images and labels from the test_loader
dataiter_test = iter(test_loader)
images, labels = next(dataiter_test)

# Move images and labels to the device
images, labels = images.to(device), labels.to(device)

# Get model predictions
with torch.no_grad():
    outputs = model(images)

# Determine predicted class
_, predicted = torch.max(outputs.data, 1)

# Plot a few images from the batch with true and predicted labels
fig = plt.figure(figsize=(12, 6))
num_images_to_display = 8 # Display up to 8 images
for i in range(num_images_to_display):
    if i >= len(images): # Ensure we don't go out of bounds if batch size is smaller than num_images_to_display
        break
    ax = fig.add_subplot(2, 4, i+1, xticks=[], yticks=[]) # 2 rows, 4 columns
    imshow_unnormalize(images[i])
    color = 'green' if predicted[i] == labels[i] else 'red'
    ax.set_title(f'True: {classes[labels[i]]}\nPred: {classes[predicted[i]]}', color=color)

plt.suptitle('Model Predictions vs. True Labels (Green: Correct, Red: Incorrect)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent suptitle overlap
plt.show()

print("Model predictions visualized successfully.")