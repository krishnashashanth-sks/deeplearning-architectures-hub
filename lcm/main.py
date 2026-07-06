import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from layers import Encoder,Decoder,Discriminator,UNetLikeStructure
from inference import sample_images
from train import train_lcm

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---  Data Loading and Preprocessing ---

# Hyperparameters for models and data
NUM_CHANNELS = 3  # RGB images
IMAGE_SIZE = 32 # CIFAR-10 image size
LATENT_CHANNELS = 4 # Channels in the latent space after encoder
TEMB_DIM = 64 # Timestep embedding dimension
BATCH_SIZE = 64

# Define transformations
train_transform_augmented = transforms.Compose([
    transforms.RandomCrop(IMAGE_SIZE, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets
full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=val_test_transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=val_test_transform)

# Split training data into training and validation sets
train_indices, val_indices = train_test_split(
    np.arange(len(full_trainset)),
    test_size=0.1, # 10% for validation
    random_state=42,
    stratify=full_trainset.targets
)

train_dataset_augmented = Subset(torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform_augmented), train_indices)
val_dataset = Subset(full_trainset, val_indices)

# Create DataLoaders
trainloader_augmented = DataLoader(train_dataset_augmented, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
valloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Augmented Training DataLoader created with {len(trainloader_augmented)} batches and {len(train_dataset_augmented)} samples.")
print(f"Validation DataLoader created with {len(valloader)} batches and {len(val_dataset)} samples.")
print(f"Test DataLoader created with {len(testloader)} batches and {len(testset)} samples.")

# ---  Loss Functions and Hyperparameters ---
mse_loss = nn.MSELoss()
bce_loss = nn.BCEWithLogitsLoss()

num_epochs = 20 # Reduced for faster execution in example, increase for better results
LEARNING_RATE = 1e-4

# Loss weighting factors
lambda_reconstruction = 1.0
lambda_consistency = 1.0
lambda_adversarial = 0.1 # Set to 0.0 to disable adversarial training

GRADIENT_CLIP_VALUE = 1.0
log_interval = 100

# Instantiate actual models
encoder = Encoder(in_channels=NUM_CHANNELS, latent_channels=LATENT_CHANNELS, channel_multipliers=[1, 2, 4]).to(device)
# The Decoder should mirror the channel multipliers of the encoder in reverse
decoder = Decoder(out_channels=NUM_CHANNELS, latent_channels=LATENT_CHANNELS, channel_multipliers=[4, 2, 1]).to(device)
# UNet input channels should match encoder's output channels after final conv_out
unet_model = UNetLikeStructure(in_channels=LATENT_CHANNELS, out_channels=LATENT_CHANNELS, temb_dim=TEMB_DIM, channel_multipliers=[1,2]).to(device) # Simplified for fewer down/up blocks
discriminator = Discriminator(num_channels=NUM_CHANNELS).to(device)

# Optimizers
optimizer_encoder = optim.AdamW(encoder.parameters(), lr=LEARNING_RATE)
optimizer_decoder = optim.AdamW(decoder.parameters(), lr=LEARNING_RATE)
optimizer_unet = optim.AdamW(unet_model.parameters(), lr=LEARNING_RATE)
optimizer_discriminator = optim.AdamW(discriminator.parameters(), lr=LEARNING_RATE)

# Train the LCM
train_lcm(
    encoder, decoder, unet_model, discriminator,
    optimizer_encoder, optimizer_decoder, optimizer_unet, optimizer_discriminator,
    trainloader_augmented, valloader, num_epochs, device,
    mse_loss, bce_loss,
    lambda_reconstruction, lambda_consistency, lambda_adversarial,
    GRADIENT_CLIP_VALUE, log_interval
)

# Load the best model for inference
print("\nLoading best model for inference...")
checkpoint = torch.load('best_lcm_model_actual.pth', map_location=device)
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])
unet_model.load_state_dict(checkpoint['unet_state_dict'])
if lambda_adversarial > 0.0 and checkpoint['discriminator_state_dict'] is not None:
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
print("Models loaded from 'best_lcm_model_actual.pth' successfully.")

# Generate and display sample images using the loaded model
print("\nGenerating sample images using loaded models...")
sample_output_loaded = sample_images(encoder, unet_model, decoder, device, num_samples=8, latent_channels=LATENT_CHANNELS, image_size=IMAGE_SIZE)

plt.figure(figsize=(10, 4))
for i in range(sample_output_loaded.shape[0]):
    plt.subplot(2, 4, i + 1)
    plt.imshow(sample_output_loaded[i])
    plt.axis('off')
plt.suptitle("Generated Sample Images from Loaded Actual Model")
plt.show()
print("Sample image generation from loaded models complete.")