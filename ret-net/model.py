from layers import *
import torch.nn as nn
from layers import RMSNorm

class RetNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        intermediate_dim: int,
        dropout: float = 0.1,
        pad_token_id: int = 0 # Default pad token ID, might need adjustment
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.pad_token_id = pad_token_id

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
        self.layers = nn.ModuleList([
            RetNetBlock(hidden_dim, num_heads, head_dim, intermediate_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm_out = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, mode: str = 'parallel', past_key_values: list[list[torch.Tensor]] = None):
        # input_ids: (batch_size, sequence_length)
        # past_key_values: list of (list of head_states) for each layer

        x = self.token_embedding(input_ids)

        if mode == 'parallel':
            for layer in self.layers:
                x, _ = layer(x, mode='parallel')

            x = self.norm_out(x)
            logits = self.lm_head(x)
            return logits, None # No states to return in parallel mode

        elif mode == 'recurrent':
            # x: (batch_size, 1, hidden_dim) as it's typically for single token generation
            # past_key_values is a list of lists: num_layers x num_heads x (head_dim, head_dim) or None
            new_past_key_values = []

            if past_key_values is None:
                # Initialize states for the first token
                # Each layer returns a list of states (one per head)
                past_key_values = [None] * self.num_layers

            for i, layer in enumerate(self.layers):
                # state for the current layer is a list of head states
                current_layer_state = past_key_values[i] if past_key_values[i] is not None else None
                x, layer_new_state = layer(x, mode='recurrent', state=current_layer_state)
                new_past_key_values.append(layer_new_state)

            x = self.norm_out(x.unsqueeze(1)).squeeze(1) # Apply RMSNorm, ensuring correct dimension for (B, H) -> (B, 1, H) -> (B, H)
            logits = self.lm_head(x) # (batch_size, vocab_size)
            return logits, new_past_key_values

        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'parallel' or 'recurrent'.")
