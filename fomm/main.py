import matplotlib.pyplot as plt
from inference import inference
from train import train_model
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import DummyVideoDataset
from model import FirstOrderMotionModel
import torch
import torch.nn as nn

# Training parameters
batch_size = 1 # Reduced batch size to conserve RAM
image_size = 128 # Reduced image resolution to conserve RAM
in_channels = 3 # RGB image
num_keypoints = 10
num_epochs = 5
learning_rate = 1e-4

# Initialize dataset and dataloader
dataset = DummyVideoDataset(num_samples=100, image_size=(image_size, image_size))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fom_model = FirstOrderMotionModel(in_channels=in_channels, num_keypoints=num_keypoints, num_filters=32).to(device)
optimizer = optim.Adam(fom_model.parameters(), lr=learning_rate)

# Using L1Loss as a basic reconstruction loss for demonstration
# In a real FOMM implementation, you'd use a combination of perceptual, adversarial, and equivariance losses.
reconstruction_loss_fn = nn.L1Loss()

train_model(fom_model,num_epochs,optimizer,reconstruction_loss_fn,dataloader,device)

# Create dummy input images (e.g., from the DummyVideoDataset)
sample_dataset = DummyVideoDataset(num_samples=1, image_size=(image_size, image_size))
sample_source_image, sample_driving_image = sample_dataset[0]

generated_image = inference(fom_model, sample_source_image, sample_driving_image, device)

# Visualize the results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Source Image
axes[0].imshow(sample_source_image.permute(1, 2, 0).numpy())
axes[0].set_title('Source Image')
axes[0].axis('off')

# Driving Image
axes[1].imshow(sample_driving_image.permute(1, 2, 0).numpy())
axes[1].set_title('Driving Image')
axes[1].axis('off')

# Generated Image
axes[2].imshow(generated_image.permute(1, 2, 0).numpy())
axes[2].set_title('Generated Image')
axes[2].axis('off')

plt.tight_layout()
plt.show()
