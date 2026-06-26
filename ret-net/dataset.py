import requests
import torch
from torch.utils.data import Dataset

# URL for Tiny Shakespeare dataset
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# Download the dataset
try:
    response = requests.get(url)
    response.raise_for_status() # Raise an exception for HTTP errors
    text_data = response.text
    print("Tiny Shakespeare dataset downloaded successfully.")
    print(f"Dataset size: {len(text_data)} characters")
    print("First 500 characters:")
    print(text_data[:500])
except requests.exceptions.RequestException as e:
    print(f"Error downloading Tiny Shakespeare dataset: {e}")
    text_data = ""

chars = sorted(list(set(text_data)))
vocab_size = len(chars)

# Create a mapping from characters to integers
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}

# Encoder and Decoder functions
encode = lambda s: [char_to_int[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([int_to_char[i] for i in l]) # decoder: take a list of integers, output a string

# Convert the entire text dataset to tensors
data = torch.tensor(encode(text_data), dtype=torch.long)

# Split data into training and validation sets
n = int(0.9 * len(data)) # 90% for training, 10% for validation
train_data = data[:n]
val_data = data[n:]

# Define the block size (sequence length for training)
block_size = 256 # A common sequence length for transformer-like models

class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        # The number of possible input sequences of length block_size
        # is len(data) - block_size
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Get a chunk of text starting from idx
        chunk = self.data[idx : idx + self.block_size + 1]

        # Input is everything up to the last token
        x = chunk[:-1]
        # Target is everything from the second token onwards (next token prediction)
        y = chunk[1:]

        return x, y

# Create dataset instances
train_dataset = TextDataset(train_data, block_size)
val_dataset = TextDataset(val_data, block_size)