from train import train_model
from dataset import TreeDataset,tree_collate_fn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from model import RecursiveNN
import torch
from layers import Node

# Define dimensions
INPUT_DIM = 50  # e.g., dimensionality of learned word embeddings
HIDDEN_DIM = 100
OUTPUT_DIM = 2  # e.g., for binary sentiment classification (positive/negative)

# Create a dummy dataset and DataLoader
NUM_SAMPLES = 100
BATCH_SIZE = 1 # Recommended batch size of 1 for simple tree processing in Recursive NNs

dataset = TreeDataset(NUM_SAMPLES, INPUT_DIM, OUTPUT_DIM)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=tree_collate_fn)

print(f"Created a TreeDataset with {len(dataset)} samples.")
print(f"DataLoader will process trees in batches of {dataloader.batch_size}.")

model = RecursiveNN(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

NUM_EPOCHS = 5

train_model(NUM_EPOCHS,dataloader,optimizer,model,criterion)

# Create a new dummy tree for inference
new_leaf_features_1 = torch.randn(1, INPUT_DIM)
new_leaf_features_2 = torch.randn(1, INPUT_DIM)

new_lf1 = Node(feature_vector=new_leaf_features_1)
new_lf2 = Node(feature_vector=new_leaf_features_2)

new_root = Node(children=[new_lf1, new_lf2])

# Set the model to evaluation mode
model.eval()

with torch.no_grad(): # Disable gradient calculations for inference
  new_root_representation = model(new_root)
  new_output_logits = model.get_output(new_root_representation)

  _, predicted_class = torch.max(new_output_logits, 1)

print(f"\nRepresentation of the new root node: {new_root_representation.shape}")
print(new_root_representation)
print(f"Logits for the new tree: {new_output_logits}")
print(f"Predicted class for the new tree: {predicted_class.item()}")