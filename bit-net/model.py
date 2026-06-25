import torch.nn as nn
from layers import BinarizedTransformerBlock
import torch

class BitNet(nn.Module):
  def __init__(self,dim,depth,num_heads,mlp_ratio=4.,qkv_bias=False,drop_rate=0.4,attn_drop_rate=0.,
               drop_path_rate=0.,norm_layer=nn.LayerNorm,num_classes=10):
    super().__init__()
    self.num_classes=num_classes
    self.num_features=self.embed_dim=dim
    self.embedding=nn.Linear(dim,dim)
    self.blocks=nn.ModuleList([
        BinarizedTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=drop_path_rate,
            norm_layer=norm_layer
        )
        for _ in range(depth)
    ])
    self.norm=norm_layer(dim)
    self.head=nn.Linear(dim,num_classes) if num_classes>0 else nn.Identity()
    self.apply(self._init_weights)
  def _init_weights(self,m):
    if isinstance(m,nn.Linear):
      nn.init.xavier_uniform_(m.weight)
      if m.bias is not None:
        nn.init.zeros_(m.bias)

  def forward_features(self, x):
    # x is assumed to be (batch_size, sequence_length, dim)
    x = self.embedding(x) # Apply initial embedding/projection
    # No activation quantization here for initial embedding output as per some BitNet variants.

    for blk in self.blocks:
      x = blk(x)
    x = self.norm(x)
    return x.mean(dim=1) # Global average pooling for sequence output

  def forward(self, x):
    x = self.forward_features(x)
    x = self.head(x)
    return x

class BitNetMNIST(BitNet):
    def __init__(self, input_dim, hidden_dim, depth, num_heads, mlp_ratio, num_classes):
        # The embedding layer now maps input_dim to hidden_dim
        super().__init__(
            dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes
        )
        # Overwrite the embedding layer to handle the flattened input
        self.embedding = nn.Linear(input_dim, hidden_dim)

    def forward_features(self, x):
        # x is assumed to be (batch_size, input_dim)
        # The original BitNet expects (batch_size, sequence_length, dim)
        # For MNIST, after flattening, it's (batch_size, 784).
        # We treat sequence_length as 1 here by unsqueezing, or the embedding layer handles it.
        # Let's make sure the embedding layer gets the correct input. If x is (B, 784), 
        # self.embedding(x) will output (B, hidden_dim).
        # The transformer blocks then expect (B, 1, hidden_dim) if we conceptualize it as a sequence of length 1
        # Or, if the transformer blocks are meant to operate on a single vector, 
        # we need to ensure the dim is consistent.
        
        # Let's adjust the forward pass to conceptually treat the input as a sequence of length 1
        # after embedding, for compatibility with the transformer blocks expecting (B, N, C)
        
        # Apply initial embedding/projection: (batch_size, input_dim) -> (batch_size, hidden_dim)
        x = self.embedding(x)
        
        # Unsqueeze to simulate a sequence of length 1: (batch_size, hidden_dim) -> (batch_size, 1, hidden_dim)
        x = x.unsqueeze(1) 

        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        # Global average pooling (over the sequence dimension, which is 1 here)
        return x.mean(dim=1) 