from torch.utils.data import Dataset
import numpy as np
import torch

class DummyDataset(Dataset):
    def __init__(self, num_samples, n_symbols, n_mel_channels, min_text_len, max_text_len, min_mel_len, max_mel_len):
        self.num_samples = num_samples
        self.n_symbols = n_symbols
        self.n_mel_channels = n_mel_channels
        self.min_text_len = min_text_len
        self.max_text_len = max_text_len
        self.min_mel_len = min_mel_len
        self.max_mel_len = max_mel_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        text_length = np.random.randint(self.min_text_len, self.max_text_len + 1)
        text = torch.randint(0, self.n_symbols, (text_length,), dtype=torch.long)

        mel_length = np.random.randint(self.min_mel_len, self.max_mel_len + 1)
        mel = torch.randn(self.n_mel_channels, mel_length, dtype=torch.float)

        return text, mel, text_length, mel_length