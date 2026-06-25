from model import BitNetMNIST
from dataset import mnist_input_dim,train_loader,test_loader
from train import train
from test import test
from inference import inference_single_image
import torch.optim as optim
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Model parameters
model_input_dim = mnist_input_dim # 784 for flattened MNIST images
model_hidden_dim = 128 # Internal dimension for transformer blocks
model_depth = 2
model_heads = 4
model_mlp_ratio = 2.
num_output_classes = 10 # 10 digits for MNIST

bitnet_mnist_model = BitNetMNIST(
    input_dim=model_input_dim,
    hidden_dim=model_hidden_dim,
    depth=model_depth,
    num_heads=model_heads,
    mlp_ratio=model_mlp_ratio,
    num_classes=num_output_classes
)

optimizer = optim.Adam(bitnet_mnist_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bitnet_mnist_model.to(device)

epochs = 10 # Number of training epochs

print("Starting BitNet training on MNIST...")
for epoch in range(1, epochs + 1):
    train(bitnet_mnist_model, device, train_loader,criterion, optimizer, epoch)
    test(bitnet_mnist_model, device, test_loader,criterion,epoch)


batch_idx, (example_data, example_targets) = next(iter(test_loader))

# Pick the first image from the batch
single_image_tensor = example_data[0] # Shape (784,)
single_image_true_label = example_targets[0].item()

# Perform inference
predicted_label = inference_single_image(bitnet_mnist_model, single_image_tensor, device)

print(f"\n--- Single Image Inference Example ---")
print(f"True Label: {single_image_true_label}")
print(f"Predicted Label: {predicted_label}")

# Visualize the single image
plt.figure(figsize=(2, 2))
plt.imshow(single_image_tensor.cpu().view(28, 28), cmap='gray', interpolation='none') # Reshape back to 28x28 for display
plt.title(f"True: {single_image_true_label}, Pred: {predicted_label}")
plt.xticks([])
plt.yticks([])
plt.show()
