from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from dataset import train_dataset,test_dataset,IMG_SIZE,test_transform
from model import AlexNet
import torch
import torch.nn as nn
from train import train
import matplotlib.pyplot as plt
from inference import predict_image

num_classes = 10 # CIFAR-10 has 10 classes

# 1. Instantiate the AlexNet model
model = AlexNet(num_classes=num_classes)

# 2. Define the loss function
criterion = nn.CrossEntropyLoss()

# 3. Define an optimizer
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 4. Check for GPU availability and move the model to the GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=64,shuffle=False)

num_epochs = 10 # Set the number of training epochs

train_losses,train_accuracies,val_losses,val_accuracies=train(num_epochs,model,train_loader,test_loader,optimizer,criterion,device)

# Plot training and validation loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("Training and validation metrics plotted successfully.")

# Define CIFAR-10 class names
cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Create a dummy image (e.g., a black square for demonstration purposes)
# In a real scenario, you would load an actual image file.
dummy_image = Image.fromarray(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))

# Make a prediction using the defined function
predicted_class, probabilities = predict_image(model, dummy_image, test_transform, device, cifar10_classes)

print(f"Predicted Class: {predicted_class}")
print("Prediction Probabilities:")
for i, prob in enumerate(probabilities):
    print(f"  {cifar10_classes[i]}: {prob:.4f}")
