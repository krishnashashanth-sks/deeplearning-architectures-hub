from model import SqueezeNet
from torch.utils.data import DataLoader
from dataset import trainset,testset
import torch.nn as nn
import torch.optim as optim
import torch
from train import train
import torchvision
from utils import imshow

model = SqueezeNet(num_classes=10) # Example for a 10-class classification problem

trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

# Instantiate model, loss, and optimizer
model = SqueezeNet(num_classes=10) # CIFAR-10 has 10 classes
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 10

train(num_epochs,trainloader,optimizer,model,criterion,device)

# Get a random test image
dataiter = iter(testloader)
images, labels = next(dataiter)

# Print images
print('GroundTruth: ', ' '.join(f'{testset.classes[labels[j]]:5s}' for j in range(4)))
imshow(torchvision.utils.make_grid(images))

# Move images to device and make a prediction
model.eval() # Ensure model is in evaluation mode
images = images.to(device)
outputs = model(images)

# Get the predicted class
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{testset.classes[predicted[j]]:5s}' for j in range(4)))