from model import ResNet
from layers import Bottleneck
from dataset import train_dataset,test_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from train import train

# 1. Set up the device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Instantiate a ResNet-50 model
def ResNet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

# Create an instance of ResNet50
model_resnet50 = ResNet50()

BATCH_SIZE = 64

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

model = model_resnet50.to(device)

#  Define the loss function
criterion = nn.CrossEntropyLoss()

#  Define an optimizer
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 1 # Example number of epochs

train(num_epochs,model,train_loader,device,optimizer,criterion,test_loader)

print("\n--- Demonstrating Single Sample Prediction ---")

# 1. Get a single sample image and its label from the test dataset
# Let's pick the first image for demonstration
single_image, single_label = test_dataset[0]

# 2. Add a batch dimension to the single image
# Models expect a batch of images, even if it's a batch of one.
# The shape will change from (C, H, W) to (1, C, H, W)
single_image_batch = single_image.unsqueeze(0)

# 3. Move the single image to the appropriate device
single_image_batch = single_image_batch.to(device)

# 4. Set the model to evaluation mode (if not already)
model.eval()

# 5. Perform inference with torch.no_grad()
with torch.no_grad():
    output = model(single_image_batch)

# 6. Interpret the output
# The output will be logits. To get probabilities, apply softmax.
probabilities = torch.softmax(output, dim=1)

# Get the predicted class (index with highest probability)
predicted_probability, predicted_class_idx = torch.max(probabilities, 1)

# Get the class names for CIFAR-10
cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

predicted_label = cifar10_classes[predicted_class_idx.item()]
actual_label = cifar10_classes[single_label]

print(f"Actual Label: {actual_label}")
print(f"Predicted Label: {predicted_label}")
print(f"Predicted Probability: {predicted_probability.item():.4f}")

print("--- Single Sample Prediction Demonstration Completed ---")