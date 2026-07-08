from PIL import Image
from torch.utils.data import Dataset
import torch

# Dummy Dataset for demonstration if no real dataset is available
class DummyDataset(Dataset):
    def __init__(self, num_samples=100, transform=None, image_size=256):
        self.num_samples = num_samples
        self.transform = transform
        self.data = [
            torch.randint(0, 256, (3, image_size, image_size), dtype=torch.uint8).permute(1, 2, 0).numpy()
            for _ in range(num_samples)
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img_array = self.data[idx]
        img = Image.fromarray(img_array)
        if self.transform:
            return self.transform(img)
        return img
