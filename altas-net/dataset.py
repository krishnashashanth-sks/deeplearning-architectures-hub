from torch.utils.data import Dataset
import torch

# Custom IrisDataset class
class IrisDataset(Dataset):
  def __init__(self, features, labels):
    self.features = torch.tensor(features.values, dtype=torch.float32)
    self.labels = torch.tensor(labels.values, dtype=torch.long)

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    return self.features[idx], self.labels[idx]
