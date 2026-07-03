import torch.optim as optim
from torchvision import datasets, transforms
from model import Net
from train import train_snn
from evaluate import evaluate_snn
import torch
from snntorch import functional as SF
from inference import inference
import matplotlib.pyplot as plt

num_inputs = 28 * 28  # MNIST image size: 28x28
num_hidden = 1000     # Number of neurons in the hidden layer
num_outputs = 10      # Number of output classes for MNIST (digits 0-9)
batch_size = 128
num_epochs = 10
num_steps = 25
beta = 0.9
threshold = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("torchvision datasets and transforms imported successfully.")

# 2. Define a transformation pipeline
transform = transforms.Compose([
    transforms.ToTensor(), # Converts a PIL Image or numpy.ndarray to a FloatTensor of shape (C, H, W) and scales it to [0.0, 1.0]
    # No normalization needed yet, as pixel values are already 0-1 range after ToTensor
])

print("Transformation pipeline defined.")

# 3. Download and load the MNIST training dataset
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

# 4. Download and load the MNIST test dataset
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)

print(f"MNIST training dataset loaded: {len(train_dataset)} samples")
print(f"MNIST test dataset loaded: {len(test_dataset)} samples")

# 5. Create a DataLoader for the training dataset
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)

# 6. Create a DataLoader for the test dataset
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True
)
# Define learning rate
learning_rate = 1e-2

# Initialize the SNN model
net = Net(num_inputs, num_hidden, num_outputs, beta, threshold).to(device)
print("SNN model initialized and moved to device.")

# Define the optimizer
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
print(f"Optimizer (Adam) initialized with learning rate: {learning_rate}")

# Define the loss function
loss_fn = SF.ce_rate_loss()
print("Loss function (snnTorch.functional.ce_rate_loss) defined.")

train_snn(net, train_loader, optimizer, loss_fn, num_epochs, num_steps, device)

print("\n--- SNN Evaluation ---")
test_accuracy = evaluate_snn(net, test_loader, num_steps, device)
print(f"Final Test Accuracy: {test_accuracy:.2f}%")

# Demonstrate inference on a sample from the test set
print("\n--- Demonstrating Inference ---")
# Get one sample from the test dataset
sample_data, sample_target = next(iter(test_loader))
single_image = sample_data[0] # Take the first image from the batch
single_label = sample_target[0].item()

# Perform inference
predicted_label = inference(net, single_image, num_steps, device)

print(f"Original label: {single_label}")
print(f"Predicted label: {predicted_label}")

# You can visualize the image if matplotlib is available
plt.imshow(single_image.squeeze().cpu().numpy(), cmap='gray')
plt.title(f"True: {single_label}, Predicted: {predicted_label}")
plt.show()