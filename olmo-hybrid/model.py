import torch.nn as nn
from layers import TransformerBlock,LinearRNNBlock

class OLMoHybridModel(nn.Module):
  def __init__(self,embed_dim,num_heads,ff_dim,num_transformer_layers,num_rnn_layers,dropout_rate=0.1):
    super().__init__()
    self.embed_dim=embed_dim
    self.transformer_layers=nn.ModuleList([
        TransformerBlock(embed_dim,num_heads,ff_dim,dropout_rate)
        for _ in range(num_transformer_layers)
    ])
    self.rnn_layers=nn.ModuleList([
        LinearRNNBlock(embed_dim)
        for _ in range(num_rnn_layers)
    ])
  def forward(self,x):
    for transformer_block in self.transformer_layers:
      x=transformer_block(x)
      h_prev=None
      for rnn_block in self.rnn_layers:
        x,h_prev=rnn_block(x,h_prev=None)
      return x