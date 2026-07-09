from torch.utils.data import Dataset
import torch

class TextDataset(Dataset):
    def __init__(self, sequences):
        # Convert list of lists to a single torch.LongTensor
        self.sequences = torch.LongTensor(sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]
