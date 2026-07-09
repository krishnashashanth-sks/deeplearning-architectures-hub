import torch
import torch.nn as nn
from layers import PositionalEncoding,RGLRULayer

class HawkModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int, # For classification/next token prediction
        max_seq_len: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len

        # 1. Input Embedding Layer
        self.embedding = nn.Embedding(vocab_size, input_dim)

        # 2. Positional Encoding
        self.positional_encoder = PositionalEncoding(input_dim, dropout, max_len=max_seq_len)

        # 3. Stacked RG-LRU Layers with LayerNorm
        self.rg_lru_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for i in range(num_layers):
            # First layer takes input_dim, subsequent layers take hidden_dim
            current_input_dim = input_dim if i == 0 else hidden_dim
            self.rg_lru_layers.append(RGLRULayer(current_input_dim, hidden_dim))
            self.norm_layers.append(nn.LayerNorm(hidden_dim))

        # 4. Final Output Layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        print(f"HawkModel initialized with {num_layers} RG-LRU layers.")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch_size, seq_len)

        # 1. Embedding
        x = self.embedding(input_ids) # (batch_size, seq_len, input_dim)

        # 2. Positional Encoding
        x = self.positional_encoder(x) # (batch_size, seq_len, input_dim)

        # 3. Stacked RG-LRU Layers
        for i, layer in enumerate(self.rg_lru_layers):
            x = layer(x) # (batch_size, seq_len, hidden_dim)
            x = self.norm_layers[i](x) # Apply LayerNorm after each RG-LRU layer

        # 4. Output Projection
        logits = self.output_layer(x) # (batch_size, seq_len, output_dim)

        return logits
