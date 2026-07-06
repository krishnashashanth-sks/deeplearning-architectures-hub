import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops,softmax

class GATLayer(MessagePassing):
  def __init__(self,in_channels,out_channels,heads=1,concat_heads=True,dropout=0.6,negative_slope=0.2,add_self_loops=True,**kwargs):
    super(GATLayer,self).__init__(node_dim=0, **kwargs)
    self.heads=heads
    self.out_channels=out_channels
    self.concat_heads=concat_heads
    self.negative_slope=negative_slope
    self.dropout=dropout
    self.add_self_loops=add_self_loops

    # Linear transformation for source and destination nodes
    # If concatenating, output dimension is heads * out_channels
    # If averaging, output dimension is out_channels (so each head outputs out_channels)
    self.lin_src = nn.Linear(in_channels, heads * out_channels, bias=False)
    self.lin_dst = self.lin_src # In GAT, usually same linear transformation for src and dst features

    # Attention mechanism parameters (a in the paper) split for src and dst
    # Correct dimensions for attention parameters: (1, heads, out_channels)
    self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
    self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))

    self.leaky_relu = nn.LeakyReLU(self.negative_slope)
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.xavier_uniform_(self.lin_src.weight)
    nn.init.xavier_uniform_(self.att_src)
    nn.init.xavier_uniform_(self.att_dst)

  def forward(self,x,edge_index):
    # Add self-loops to the edge_index
    if self.add_self_loops:
      edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

    # Linear transformation of node features for each head
    # x_src and x_dst will have shape [num_nodes, heads, out_channels]
    x_src = self.lin_src(x).view(-1, self.heads, self.out_channels)
    x_dst = self.lin_dst(x).view(-1, self.heads, self.out_channels)

    # Propagate messages and compute aggregated features
    # The propagate function will call message, aggregate, and update
    out = self.propagate(edge_index, x=(x_src,x_dst), size=None)

    # Combine outputs from different attention heads
    if self.concat_heads:
      # Concatenate heads: [num_nodes, heads * out_channels]
      out = out.view(-1, self.heads * self.out_channels)
    else:
      # Average heads: [num_nodes, out_channels]
      out = out.mean(dim=1) # Mean along the heads dimension

    # Removed F.elu activation from here to allow external control
    return out

  def message(self,x_i,x_j,edge_index_i,size_i):
    # x_i, x_j are now [num_edges, heads, out_channels]
    # att_src, att_dst are [1, heads, out_channels]

    # Calculate raw attention scores for source and destination parts
    # (x_j * self.att_src).sum(dim=-1, keepdim=True) will be [num_edges, heads, 1]
    alpha_src = (x_i * self.att_dst).sum(dim=-1, keepdim=True) # Note: x_i uses att_dst for target
    alpha_dst = (x_j * self.att_src).sum(dim=-1, keepdim=True) # Note: x_j uses att_src for source

    # Combine scores and apply LeakyReLU
    alpha = self.leaky_relu(alpha_src + alpha_dst)

    # Apply softmax to normalize attention coefficients over neighborhoods for each head
    # alpha shape is [num_edges, heads, 1]
    alpha = softmax(alpha, edge_index_i, num_nodes=size_i)

    # Apply dropout to attention coefficients
    alpha = F.dropout(alpha, p=self.dropout, training=self.training)

    # Weight neighbor features by attention coefficients
    # x_j is [num_edges, heads, out_channels], alpha is [num_edges, heads, 1]
    return x_j * alpha

  def update(self,aggr_out):
    # aggr_out is of shape [num_nodes, heads, out_channels]
    return aggr_out