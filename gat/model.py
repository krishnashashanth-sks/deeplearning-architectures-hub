import torch.nn as nn
import torch.nn.functional as F
from layers import GATLayer

class GAT(nn.Module):
  def __init__(self,in_channels,hidden_channels,out_channels,num_heads=8,dropout=0.6):
    super(GAT,self).__init__()
    # First GAT layer: takes original features, outputs 'hidden_channels' features per head, concatenates heads
    self.conv1=GATLayer(
        in_channels=in_channels,
        out_channels=hidden_channels, # Correct: each head outputs hidden_channels
        heads=num_heads,
        concat_heads=True, # Concatenate outputs from all heads
        dropout=dropout
    )
    # Second GAT layer: takes concatenated output from conv1 as input, outputs final classification dimension
    # It typically uses one head (or averages multiple heads) for the final layer.
    self.conv2=GATLayer(
        in_channels=num_heads*hidden_channels, # Input is the concatenated output of conv1
        out_channels=out_channels, # Output dimension is the number of classes
        heads=1, # Use a single attention head for the final layer
        concat_heads=False, # Average (or just pass through if heads=1) for the final output
        dropout=dropout
    )
    self.dropout_rate=dropout

  def forward(self,x,edge_index):
    x=F.dropout(x,p=self.dropout_rate,training=self.training)
    x=self.conv1(x,edge_index) # Output shape: [num_nodes, num_heads * hidden_channels]
    x=F.elu(x) # Apply activation after the first GAT layer
    x=F.dropout(x,p=self.dropout_rate,training=self.training)
    return self.conv2(x,edge_index) # Output shape: [num_nodes, out_channels], no activation here for logits