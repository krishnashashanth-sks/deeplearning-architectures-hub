import torchvision.transforms as transforms
import torchvision
import torch

# Original image size: 28x28 = 784 pixels
mnist_input_dim = 28 * 28

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)), # MNIST dataset normalization
    transforms.Lambda(lambda x: x.view(-1)) # Flatten the 28x28 image into a 784 vector
])

# Download and load the training data
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Download and load the test data
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

print("MNIST dataset loaded and DataLoaders created.")
