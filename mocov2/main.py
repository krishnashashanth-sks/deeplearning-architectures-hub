import torch
from torch.utils.data import DataLoader
import os
from transform import get_moco_transform
from dataset import DummyDataset
from model import MoCo
from layers import BaseEncoder
from losses import InfoNCELoss
from train import train_moco

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
image_size = 224
batch_size = 32
num_epochs = 5 # Set a higher number for real training
learning_rate = 0.03
momentum = 0.9
weight_decay = 1e-4
moco_dim = 128
moco_queue_size = 65536
moco_momentum = 0.999
moco_temperature = 0.07
moco_mlp_hidden_dim = 2048
# Set your dataset directory here, e.g., 'path/to/your/imagenet/train'
# If None or path doesn't exist, DummyDataset will be used.
dataset_directory = None 
# dataset_directory = '/path/to/your/dataset'

# 1 Setup DataLoader

data_dir=dataset_directory,
batch_size=batch_size,
num_workers=os.cpu_count(), # Use all available CPU cores for data loading
image_size=image_size

moco_transform = get_moco_transform(image_size)

if data_dir and os.path.exists(data_dir):
    # For a real implementation, use ImageFolder or a custom dataset
    from torchvision.datasets import ImageFolder
    dataset = ImageFolder(root=data_dir, transform=moco_transform)
    print(f"Using ImageFolder from: {data_dir} with {len(dataset)} samples.")
else:
    # Fallback to DummyDataset if data_dir is not provided or doesn't exist
    print("No real dataset directory provided or found, using DummyDataset.")
    dataset = DummyDataset(num_samples=200, transform=moco_transform, image_size=image_size)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True # Important for MoCo to maintain consistent batch sizes and queue updates
)

# 2. Initialize MoCo model
moco_model = MoCo(
    BaseEncoder,
    dim=moco_dim,
    K=moco_queue_size,
    m=moco_momentum,
    T=moco_temperature,
    mlp_hidden_dim=moco_mlp_hidden_dim
).to(device)

# 3. Initialize Loss function and Optimizer
contrastive_loss_fn = InfoNCELoss()
optimizer = torch.optim.SGD(
    moco_model.parameters(), # Optimize all MoCo parameters (query encoder + projector)
    lr=learning_rate,
    momentum=momentum,
    weight_decay=weight_decay
)
# Note: MoCo paper typically uses specific learning rate schedules (e.g., cosine) and 
# often separates LR for encoder and projector or uses a base LR for the encoder 
# and a scaled LR for the projector, but this is a simplified setup.
# optimizer.add_param_group({'params': moco_model.projector_q.parameters(), 'lr': learning_rate * scale})

print("\nMoCo v2 model initialized:")
print(moco_model)

# 4. Run Training Loop
train_moco(moco_model, dataloader, optimizer, contrastive_loss_fn, num_epochs, device)

# --- Conceptual Evaluation and Inference --- 
# After pre-training, you would typically save the encoder and train a linear classifier.
print("\n--- Conceptual Evaluation & Inference (Feature Extraction) ---")
moco_model.eval() # Set model to evaluation mode
with torch.no_grad():
    # Example: Extract features for a single dummy image
    dummy_eval_image = torch.randn(1, 3, image_size, image_size).to(device)
    features = moco_model.encoder_q(dummy_eval_image) # Get features from the base encoder
    print(f"Features for a dummy evaluation image (shape): {features.shape}")
    print("\nTo evaluate, you would typically: ")
    print("1. Freeze `moco_model.encoder_q`.")
    print("2. Train a new linear classifier on top of its extracted features for a downstream task (e.g., ImageNet classification). ")
    print("3. Evaluate the classifier's accuracy on a validation set.")
    
    # Example: Get features for a batch of new images
    dummy_inference_batch = torch.randn(5, 3, image_size, image_size).to(device)
    inference_features = moco_model.encoder_q(dummy_inference_batch)
    print(f"\nInferred features for a batch of images (shape): {inference_features.shape}")
    print("These features can be used for tasks like image retrieval, clustering, or passed through a trained linear classifier for prediction.")
