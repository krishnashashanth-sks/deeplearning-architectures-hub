import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Initialize gain with ones, which will be learnable
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # Root Mean Square calculation
        # rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) is more numerically stable
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x) * self.weight

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, intermediate_dim)
        self.gelu = nn.GELU()
        self.w2 = nn.Linear(intermediate_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(self.gelu(self.w1(x))))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, seq_len=2048):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len = seq_len

    def forward(self, x, seq_dim=1):
        t = torch.arange(x.shape[seq_dim], device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Reshape for broadcasting with x
        # Add singleton dimensions for batch and head if x is (batch, seq_len, head_dim) or (batch, num_heads, seq_len, head_dim)
        # If x is (batch, num_heads, seq_len, head_dim), emb needs to be (1, 1, seq_len, head_dim)
        if x.ndim == 3: # (batch, seq_len, dim)
            emb = emb.view(1, -1, emb.shape[-1])
        elif x.ndim == 4: # (batch, num_heads, seq_len, dim_per_head)
            emb = emb.view(1, 1, -1, emb.shape[-1])

        # Apply rotary embedding
        cos = emb.cos()
        sin = emb.sin()

        # Apply rotation using complex numbers or explicitly
        # x_rot = x * cos + self.rotate_half(x) * sin

        x_real, x_imag = x.float().chunk(2, dim=-1)
        cos_real, cos_imag = cos.float().chunk(2, dim=-1) # These will be the same actually
        sin_real, sin_imag = sin.float().chunk(2, dim=-1) # These will be the same actually

        # Apply RoPE
        # (x_real * cos - x_imag * sin), (x_real * sin + x_imag * cos)
        rotated_x_real = x_real * cos_real - x_imag * sin_real
        rotated_x_imag = x_real * sin_real + x_imag * cos_real

        return torch.cat((rotated_x_real, rotated_x_imag), dim=-1).type_as(x)

    def rotate_half(self, x):
        x = x.view(x.shape[:-1] + (-1, 2))
        x1, x2 = x.unbind(dim=-1)
        return torch.cat((-x2, x1), dim=-1).view(x.shape[:-1] + (-1,)) # Reshape back


class Retention(nn.Module):
    def __init__(self, hidden_dim: int, head_dim: int, gamma: float = 0.96875):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim

        self.q_proj = nn.Linear(hidden_dim, head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, head_dim, bias=False)

        self.rope = RotaryPositionalEmbedding(head_dim) # RoPE applies to Q and K

        # Exponential decay parameter. Fixed for this single head as per RetNet paper for gamma_0.
        self.gamma = gamma

    def _get_decay_matrix(self, seq_len: int, device: torch.device):
        # Create a lower triangular mask for causality
        indices = torch.arange(seq_len, device=device)
        mask = indices.unsqueeze(0) >= indices.unsqueeze(1) # (seq_len, seq_len)

        # Calculate decay factors: gamma^(i-j)
        decay_factors = torch.pow(self.gamma, (indices.unsqueeze(0) - indices.unsqueeze(1)).float().abs())

        decay_matrix = decay_factors * mask.float()

        # Normalization term for parallel mode as in RetNet paper
        # sum_{k=0}^{i} gamma^k = (1 - gamma^(i+1)) / (1 - gamma)
        i_plus_1 = torch.arange(1, seq_len + 1, device=device).float()
        denominator = (1.0 - torch.pow(self.gamma, i_plus_1)) / (1.0 - self.gamma)

        denominator = denominator.unsqueeze(-1)
        decay_matrix = decay_matrix / (denominator + 1e-8) # Add small epsilon for numerical stability

        return decay_matrix

    def forward(self, x: torch.Tensor, mode: str = 'parallel', state: torch.Tensor = None):
        # x: (batch_size, sequence_length, hidden_dim) for parallel
        # x: (batch_size, 1, hidden_dim) for recurrent (current token)

        q = self.q_proj(x) # (batch_size, S, head_dim)
        k = self.k_proj(x) # (batch_size, S, head_dim)
        v = self.v_proj(x) # (batch_size, S, head_dim)

        # Apply Rotary Positional Embedding to Q and K
        q = self.rope(q, seq_dim=1)
        k = self.rope(k, seq_dim=1)

        if mode == 'parallel':
            seq_len = x.shape[1]
            decay_matrix = self._get_decay_matrix(seq_len, x.device) # (seq_len, seq_len)

            qk_scores = (q @ k.transpose(-2, -1)) # (batch_size, seq_len, seq_len)

            # Apply decay matrix (broadcasting handles batch dimension)
            retention_scores = qk_scores * decay_matrix.unsqueeze(0) # (batch_size, seq_len, seq_len)

            output = retention_scores @ v # (batch_size, seq_len, head_dim)
            return output, None # No state to return for parallel mode

        elif mode == 'recurrent':
            # x is (batch_size, 1, hidden_dim)
            # q, k, v are (batch_size, 1, head_dim)

            if state is None:
                # Initialize state (S_0) for the first token
                state = torch.zeros(x.shape[0], self.head_dim, self.head_dim, device=x.device, dtype=x.dtype)

            # Current K and V (remove seq_len=1 dimension for matrix multiplication)
            current_k = k.squeeze(1) # (batch_size, head_dim)
            current_v = v.squeeze(1) # (batch_size, head_dim)
            current_q = q.squeeze(1) # (batch_size, head_dim)

            # Compute new state: S_t = gamma * S_{t-1} + K_t^T @ V_t
            kv_product = current_k.unsqueeze(-1) @ current_v.unsqueeze(-2) # (batch_size, head_dim, head_dim)
            new_state = self.gamma * state + kv_product

            # Compute output: O_t = Q_t @ S_t
            output = (current_q.unsqueeze(1) @ new_state) # (batch_size, 1, head_dim)

            return output, new_state # Return output and new state
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'parallel' or 'recurrent'.")

class MultiHeadRetention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.output_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)
        self.dropout_layer = nn.Dropout(dropout)

        self.retention_heads = nn.ModuleList()
        # Generate gamma values for each head, typically 1 - 2**(-5 - 0.5 * j) as in some implementations
        for i in range(num_heads):
            gamma_i = 1 - 2**(-5 - 0.5 * i)
            # Each Retention head takes the full hidden_dim input and projects it to its head_dim
            self.retention_heads.append(Retention(hidden_dim=hidden_dim, head_dim=head_dim, gamma=gamma_i))

    def forward(self, x: torch.Tensor, mode: str = 'parallel', state: list[torch.Tensor] = None):
        # x: (batch_size, sequence_length, hidden_dim) for parallel
        # x: (batch_size, 1, hidden_dim) for recurrent (current token)

        outputs = []
        new_states = []

        if mode == 'parallel':
            for i in range(self.num_heads):
                head_output, _ = self.retention_heads[i](x, mode='parallel')
                outputs.append(head_output)

            combined_output = torch.cat(outputs, dim=-1) # (batch_size, seq_len, num_heads * head_dim)
            output = self.output_proj(combined_output)
            return self.dropout_layer(output), None # No state for parallel mode

        elif mode == 'recurrent':
            # For recurrent mode, x is (batch_size, 1, hidden_dim)
            # State management needs to be per head.
            if state is None:
                # Initialize state for all heads if not provided
                state = [None] * self.num_heads

            recurrent_outputs = []
            recurrent_states_out = []
            for i in range(self.num_heads):
                current_head_state = state[i] # Pass the state for the current head
                head_output, new_head_state = self.retention_heads[i](x, mode='recurrent', state=current_head_state)
                recurrent_outputs.append(head_output)
                recurrent_states_out.append(new_head_state)

            combined_output = torch.cat(recurrent_outputs, dim=-1) # (batch_size, 1, num_heads * head_dim)
            output = self.output_proj(combined_output)
            return self.dropout_layer(output), recurrent_states_out # Return output and list of new states
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'parallel' or 'recurrent'.")

class RetNetBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, intermediate_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim)
        self.retention = MultiHeadRetention(hidden_dim, num_heads, head_dim, dropout)
        self.norm2 = RMSNorm(hidden_dim)
        self.ffn = FeedForwardNetwork(hidden_dim, intermediate_dim, dropout)

    def forward(self, x: torch.Tensor, mode: str = 'parallel', state: list[torch.Tensor] = None):
        # x: (batch_size, sequence_length, hidden_dim) for parallel
        # x: (batch_size, 1, hidden_dim) for recurrent

        # Retention block
        norm_x = self.norm1(x)
        if mode == 'parallel':
            retention_output, _ = self.retention(norm_x, mode='parallel')
            x = x + retention_output # Residual connection
            new_state = None
        elif mode == 'recurrent':
            # For recurrent mode, x is (batch_size, 1, hidden_dim)
            retention_output, new_state = self.retention(norm_x, mode='recurrent', state=state)
            x = x + retention_output.squeeze(1) # Residual connection, output is (B, 1, H)
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'parallel' or 'recurrent'.")

        # FFN block
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output # Residual connection

        return x, new_state # Return output and state (if recurrent)
