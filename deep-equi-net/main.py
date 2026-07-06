import torch.optim as optim
from model import DEQClassifier
import torch
import torch.nn as nn
from train import train_model
from inference import predict_image
import matplotlib.pyplot as plt
from visualize import imshow

# 1. Initialize the DEQClassifier model
input_size = 28 * 28
hidden_dim = 256
num_classes = 10
max_iter_deq = 50
tol_deq = 1e-4

model = DEQClassifier(input_size=input_size,
                      hidden_dim=hidden_dim,
                      num_classes=num_classes,
                      max_iter=max_iter_deq,
                      tol=tol_deq)

# 4. Determine device and move model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model moved to device: {device}")

# 2. Define the loss function
criterion = nn.CrossEntropyLoss()

# 3. Define the optimizer
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print(f"Optimizer (Adam) defined with learning rate: {learning_rate}")

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

# 1. Choose a suitable dataset (FashionMNIST)
# 2. Define data transformations
# For FashionMNIST (grayscale images), mean and std for normalization are typically 0.5
transform = transforms.Compose([
    transforms.ToTensor(), # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    transforms.Normalize((0.5,), (0.5,)) # Normalize with mean and standard deviation
])

# Load the training and test datasets
train_dataset_full = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# 3. Create training, validation, and test datasets
# Split the full training dataset into training and validation sets
train_size = int(0.8 * len(train_dataset_full))
val_size = len(train_dataset_full) - train_size
train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

# Define batch size
batch_size = 32

# 4. Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
print(f"Batch size: {batch_size}")

num_epochs = 10 # Define the number of training epochs

train_losses,val_losses,train_accuracies,val_accuracies=train_model(num_epochs,model,train_loader,val_loader,optimizer,criterion,device)

# 1. Visualization of Losses
print("Generating loss visualization...")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()

print("Loss visualization complete.")

# Get a sample image from the test set
images, labels = next(iter(test_loader))

# Define class names for FashionMNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Select the first image from the batch for demonstration
sample_image = images[0]
sample_label = labels[0]

# Perform inference
predicted_class = predict_image(model, sample_image, device, class_names)

print(f"Actual Label: {class_names[sample_label.item()]}")
print(f"Predicted Label: {predicted_class}")

imshow(torchvision.utils.make_grid(sample_image.cpu()))

