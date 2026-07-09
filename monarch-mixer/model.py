from layers import M2Block

import torch.nn as nn
import torch.nn.functional as F
class M2Model(nn.Module):
  def __init__(self,vocab_size,embed_dim,num_layers,kernel_size,
               num_blocks_conv,intermediate_dim_conv,
               num_blocks_mlp,intermediate_dim_mlp, # Corrected parameter name
               dropout_rate,num_classes):
    super(M2Model,self).__init__()
    self.embed_dim=embed_dim
    self.num_layers=num_layers
    self.num_classes=num_classes
    self.embedding=nn.Embedding(vocab_size,embed_dim)
    self.m2_blocks=nn.ModuleList([
        M2Block(embed_dim=embed_dim,
                kernel_size=kernel_size,
                num_blocks_conv=num_blocks_conv,
                intermediate_dim_conv=intermediate_dim_conv,
                num_blocks_mlp=num_blocks_mlp,
                intermediate_dim_mlp=intermediate_dim_mlp,
                dropout_rate=dropout_rate)
        for _ in range(num_layers)
    ])
    self.final_norm=nn.LayerNorm(embed_dim)
    self.pool=nn.AdaptiveAvgPool1d(1) # Corrected typo
    self.classifier=nn.Linear(embed_dim,num_classes)
  def forward(self,x):
    x=self.embedding(x)
    x=x.permute(0,2,1)
    for block in self.m2_blocks:
      x=block(x)
    x=self.final_norm(x.permute(0,2,1)).permute(0,2,1)
    x=self.pool(x)
    x=x.flatten(1)
    return self.classifier(x)