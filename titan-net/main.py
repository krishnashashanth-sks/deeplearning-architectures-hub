import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from model import TitanNet50
from train import train_model
from visualize import imshow

# --- 1. Data Preparation ---

# Define data transformations for CIFAR-10
# For training, include data augmentation and normalization
# For testing, only include normalization

# CIFAR-10 mean: (0.4914, 0.4822, 0.4465)
# CIFAR-10 std: (0.2471, 0.2435, 0.2616)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
])

print("Defined data transformations for CIFAR-10.")

# Load CIFAR-10 datasets
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

print("CIFAR-10 training and test datasets loaded.")

# Create DataLoaders
batch_size = 128

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# Define the 10 classes in CIFAR-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print("DataLoaders for CIFAR-10 created.")

#2. Instantiate the model
num_classes = len(classes)
model = TitanNet50(num_classes=num_classes)

# Move model to appropriate device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Print a summary of the model
try:
    from torchinfo import summary
    summary(model, input_size=(1, 3, 32, 32))
except ImportError:
    print("torchinfo not installed. Install with 'pip install torchinfo' for detailed summary.")
    print(model)

print(f"TitanNet model instantiated with {num_classes} classes and moved to {device}.")

# --- 3. Set up Training Infrastructure ---

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

# Define the learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

print("Loss function, optimizer, and learning rate scheduler defined.")

# --- 4. Use Training Loop ---


num_epochs = 5 # Set a smaller number of epochs for demonstration
train_losses,train_accuracies,val_losses,val_accuracies = train_model(num_epochs,model,trainloader,testloader,optimizer,criterion,scheduler,device)

print("Metric lists initialized and number of epochs defined.")

# --- 5. Evaluate Model Performance (Inference) ---

# Plotting training and validation metrics
plt.figure(figsize=(12, 5))

epochs_completed = len(train_losses)

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs_completed + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs_completed + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs_completed + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, epochs_completed + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("Training and validation loss/accuracy plots generated.")

# Evaluate model on the test set
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

final_test_accuracy = 100 * correct / total
print(f'Accuracy of the model on the 10000 test images: {final_test_accuracy:.2f}%')

# Visualize sample predictions
dataiter = iter(testloader)
images, labels = next(dataiter)

images = images.to(device)
labels = labels.to(device)

outputs = model(images)
_, predicted = torch.max(outputs.data, 1)

images = images.cpu()
labels = labels.cpu()
predicted = predicted.cpu()

fig = plt.figure(figsize=(10, 8))
for i in range(min(5, len(images))):
    ax = fig.add_subplot(1, 5, i+1, xticks=[], yticks=[])
    imshow(images[i])
    ax.set_title(f"True: {classes[labels[i]]}\nPred: {classes[predicted[i]]}", color=("green" if predicted[i] == labels[i] else "red"))
plt.tight_layout()
plt.show()

print("Sample predictions displayed.")
