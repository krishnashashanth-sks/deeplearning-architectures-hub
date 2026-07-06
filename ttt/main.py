import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import torch
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import CustomCNN
from dataset import CIFAR10TestSSL
from transform import RotationTransform
from evaluate import adapt_and_evaluate_with_entropy,adapt_and_evaluate
from train import train_model
from losses import entropy_loss_fn

# Define standard transformations for supervised training data
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 3. Define standard transformations for the test data
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Define post-rotation transforms (ToTensor and Normalize)
post_rotation_img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 5. Load a suitable dataset (CIFAR-10)
# Supervised training dataset
train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform_train
)

# Test dataset for evaluation (without self-supervision)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform_test
)


test_dataset_for_ttt = CIFAR10TestSSL(
    root='./data',
    train=False,
    download=True,
    transform=transform_test, # For the classification task (standard test transforms)
    rotation_transform=RotationTransform(), # For generating rotated image and its label
    post_rotation_transform=post_rotation_img_transform # For ToTensor and Normalize on the rotated image
)

# 6. Create PyTorch DataLoader instances
batch_size = 64
num_workers = 2

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers
)

test_loader_for_ttt = DataLoader(
    test_dataset_for_ttt,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers
)

print("Datasets and DataLoaders prepared successfully.")
print(f"Training dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
print(f"Test dataset for TTT size: {len(test_dataset_for_ttt)}")

# 1. Set up the device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Instantiate the CustomCNN model and move it to the selected device
model = CustomCNN(num_classes=10, num_ss_classes=4) # CIFAR-10 has 10 classes
model.to(device)
print("CustomCNN model instantiated and moved to device.")

# 3. Define the optimizer and learning rate
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print(f"Optimizer: Adam, Learning Rate: {learning_rate}")

# 4. Define the loss function for the classification task
criterion = nn.CrossEntropyLoss()
print("Loss function: CrossEntropyLoss")

# 5. Implement a training loop
num_epochs = 10 # You can adjust this number
train_model(num_epochs,model,train_loader,optimizer,criterion,device)

# 6. After training, save the state dictionary of the trained model
model_save_path = 'supervised_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Trained model weights saved to {model_save_path}")

# 1. Instantiate a new CustomCNN model and load the weights
# This model will be loaded for each batch, so we don't instantiate it here
# but within the loop as per instruction 5.b.

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Define adaptation parameters
adapt_lr = 0.001
adapt_steps = 1
print(f"Adaptation Learning Rate: {adapt_lr}, Adaptation Steps per batch: {adapt_steps}")

# 3. Initialize the self-supervised loss function
ss_criterion = nn.CrossEntropyLoss()
print("Self-supervised loss function: CrossEntropyLoss")
all_preds,all_targets=adapt_and_evaluate(test_loader_for_ttt,CustomCNN,adapt_steps,adapt_lr,ss_criterion,device)
print("Finished test-time adaptation and evaluation.")

# Calculate final accuracy
accuracy = accuracy_score(all_targets, all_preds)
print(f"Overall Test-Time Adapted Model Accuracy: {accuracy:.4f}")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Evaluating the original supervised model on the test set (without TTT adaptation)...")

# Instantiate the CustomCNN model
model_supervised = CustomCNN(num_classes=10, num_ss_classes=4)

# Load the previously saved supervised weights
model_supervised.load_state_dict(torch.load('supervised_model.pth'))
model_supervised.to(device)

# Set model to evaluation mode
model_supervised.eval()

all_preds_supervised = []
all_targets_supervised = []

with torch.no_grad(): # Disable gradient calculations for evaluation
    for inputs, labels in test_loader: # Use the regular test_loader without SS transformations
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass for classification
        cls_outputs, _ = model_supervised(inputs)
        preds = torch.argmax(cls_outputs, dim=1)

        all_preds_supervised.extend(preds.cpu().numpy())
        all_targets_supervised.extend(labels.cpu().numpy())

# Calculate accuracy for the supervised model
accuracy_supervised = accuracy_score(all_targets_supervised, all_preds_supervised)
print(f"Original Supervised Model Accuracy on Test Set: {accuracy_supervised:.4f}")

# Print TTT Adapted Model Accuracy for comparison (from previous step)
print(f"Test-Time Adapted Model Accuracy (with TTT): {accuracy:.4f}")

# You can also add further analysis here, e.g., difference in accuracy
accuracy_diff = accuracy - accuracy_supervised
print(f"Improvement from TTT: {accuracy_diff:.4f}")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Define adaptation parameters
adapt_lr = 0.001
adapt_steps = 1
entropy_weight = 0.1 # New hyperparameter for entropy loss
print(f"Adaptation Learning Rate: {adapt_lr}, Adaptation Steps per batch: {adapt_steps}")
print(f"Entropy Loss Weight: {entropy_weight}")

# 3. Initialize the self-supervised loss function
ss_criterion = nn.CrossEntropyLoss()
print("Self-supervised loss function: CrossEntropyLoss")

# 4. Prepare for evaluation by initializing empty lists

all_preds_entropy_ttt,all_targets_entropy_ttt=adapt_and_evaluate_with_entropy(test_loader_for_ttt,CustomCNN,adapt_lr,adapt_steps,entropy_loss_fn,ss_criterion,entropy_weight,device)

# Calculate final accuracy for TTT with entropy minimization
accuracy_entropy_ttt = accuracy_score(all_targets_entropy_ttt, all_preds_entropy_ttt)
print(f"Overall Test-Time Adapted Model (with Entropy) Accuracy: {accuracy_entropy_ttt:.4f}")

# Print TTT Adapted Model Accuracy (from previous step, for comparison)
print(f"Test-Time Adapted Model (without Entropy) Accuracy: {accuracy:.4f}")

# Print Original Supervised Model Accuracy (for comparison)
print(f"Original Supervised Model Accuracy on Test Set: {accuracy_supervised:.4f}")
