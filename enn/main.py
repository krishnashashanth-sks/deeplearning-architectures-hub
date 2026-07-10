import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import ENNModel
from train import train
from test import test

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize(32), # Resize MNIST images to 32x32, as expected by the ENNModel's pooling layers
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # Standard MNIST normalization
])

# Load the MNIST dataset
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# Create DataLoaders
batch_size = 64 # Use a reasonable batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"\nTraining dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")


input_channels = 1
num_classes = 10
model = ENNModel(input_channels, num_classes).to(device)

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1# You can adjust the number of epochs

print("Starting training...")

for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, optimizer, epoch,criterion)
    test(model, device, test_loader,criterion)

print("Training complete.")

print("Running final evaluation on the test set...")
final_test_loss, final_test_accuracy = test(model, device, test_loader)
print(f"Final Test Accuracy: {100. * final_test_accuracy:.2f}%")
print(f"Final Test Loss: {final_test_loss:.4f}")

# Set the model to evaluation mode
model.eval()

# Select a few sample images from the test set
sample_images = []
sample_labels = []
num_samples_to_show = 5

# Iterate through the test_loader to get samples
for images, labels in test_loader:
    for i in range(min(num_samples_to_show, len(images))):
        sample_images.append(images[i])
        sample_labels.append(labels[i])
    if len(sample_images) >= num_samples_to_show:
        break

print(f"Selected {len(sample_images)} sample images for equivariance demonstration.")

rotation_angles = [0, 90, 180, 270] # C4 group rotations

print("Demonstrating C4 equivariance...")

for i in range(len(sample_images)):
    image = sample_images[i]
    true_label = sample_labels[i].item()

    fig, axes = plt.subplots(1, len(rotation_angles) + 1, figsize=(20, 4))
    fig.suptitle(f"Sample {i+1}: True Label = {true_label}", fontsize=16)

    # Move image to device and add batch dimension
    image_on_device = image.unsqueeze(0).to(device) # Add batch dimension

    # --- Original Image Prediction ---
    with torch.no_grad():
        output_original = model(image_on_device)
    pred_original = output_original.argmax(dim=1, keepdim=True).item()

    # Display original image
    axes[0].imshow(image.squeeze().cpu().numpy(), cmap='gray')
    axes[0].set_title(f"Original (Pred: {pred_original})")
    axes[0].axis('off')

    # --- Rotated Images Predictions ---
    for j, angle in enumerate(rotation_angles):
        # Apply rotation using torchvision.transforms.functional
        rotated_image = TF.rotate(image, angle=angle)
        rotated_image_on_device = rotated_image.unsqueeze(0).to(device)

        with torch.no_grad():
            output_rotated = model(rotated_image_on_device)
        pred_rotated = output_rotated.argmax(dim=1, keepdim=True).item()

        # Display rotated image
        axes[j+1].imshow(rotated_image.squeeze().cpu().numpy(), cmap='gray')
        axes[j+1].set_title(f"Rotated {angle}° (Pred: {pred_rotated})")
        axes[j+1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()

    print(f"Observation for Sample {i+1} (True: {true_label}):")
    print(f"  Original Prediction: {pred_original}")
    for j, angle in enumerate(rotation_angles):
        rotated_image = TF.rotate(image, angle=angle)
        rotated_image_on_device = rotated_image.unsqueeze(0).to(device)
        with torch.no_grad():
            output_rotated = model(rotated_image_on_device)
        pred_rotated = output_rotated.argmax(dim=1, keepdim=True).item()
        print(f"  Rotated {angle}° Prediction: {pred_rotated}")
    print("--------------------------------------------------")

print("Equivariance demonstration complete.")