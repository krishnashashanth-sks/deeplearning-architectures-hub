from torch.utils.data import Dataset
import torch
from layers import Node

class TreeDataset(Dataset):
  def __init__(self, num_samples, input_dim, output_dim):
    self.data = []
    for _ in range(num_samples):
      # Create a random tree structure and assign a random label
      leaf_features_1 = torch.randn(1, input_dim)
      leaf_features_2 = torch.randn(1, input_dim)
      leaf_features_3 = torch.randn(1, input_dim)
      leaf_features_4 = torch.randn(1, input_dim)

      lf1 = Node(feature_vector=leaf_features_1)
      lf2 = Node(feature_vector=leaf_features_2)
      lf3 = Node(feature_vector=leaf_features_3)
      lf4 = Node(feature_vector=leaf_features_4)

      node1 = Node(children=[lf1, lf2])
      node2 = Node(children=[lf3, lf4])

      root_node = Node(children=[node1, node2])
      target_label = torch.randint(0, output_dim, (1,))
      self.data.append((root_node, target_label))

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]

# Custom collate_fn for handling tree structures
def tree_collate_fn(batch):
  # For this basic R-NN, we process trees one-by-one. Batch size will be 1 for each tree.
  # The labels can be batched if needed, but for simplicity, we'll keep them individual here.
  trees = [item[0] for item in batch]
  labels = torch.stack([item[1] for item in batch])
  return trees, labels