import torch
import torch.nn as nn
import torch.nn.functional as F

class JambaModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        d_state,
        d_conv,
        expand,
        num_experts,
        top_k,
        max_seq_len,
        dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # 1. Input Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        # 2. Positional Embeddings (fixed for simplicity, can be learned)
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)
        self.dropout_embed = nn.Dropout(dropout)

        # 3. Stack of JambaLayers
        self.layers = nn.ModuleList([
            JambaLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # 4. Final Layer Normalization
        self.final_norm = nn.LayerNorm(d_model)

        # 5. Prediction Head
        self.prediction_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape

        # 1. Apply token embeddings
        token_embeds = self.token_embeddings(input_ids)

        # 2. Add positional embeddings
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embeddings(positions)

        # Combine token and positional embeddings
        x = self.dropout_embed(token_embeds + position_embeds)

        # Create attention mask for Transformer blocks
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(1)
        else:
            mask = None

        # Initialize lists to store expert statistics from each layer
        all_expert_loads = []
        all_expert_probabilities = []

        # 3. Pass through JambaLayers
        for layer in self.layers:
            x, expert_load, expert_probability = layer(x, mask=mask)
            all_expert_loads.append(expert_load)
            all_expert_probabilities.append(expert_probability)

        # 4. Apply final Layer Normalization
        x = self.final_norm(x)

        # 5. Pass through prediction head
        logits = self.prediction_head(x)

        return logits, all_expert_loads, all_expert_probabilities
