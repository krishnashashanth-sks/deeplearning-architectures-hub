import torch
import torch.nn as nn
from torch_geometric.utils import scatter # Explicitly importing scatter for EdgeConv

class EdgeConv(nn.Module):
  def __init__(self,in_channels,out_channels):
    super(EdgeConv,self).__init__()
    self.mlp=nn.Sequential(
        nn.Linear(2*in_channels,out_channels),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
        nn.Linear(out_channels,out_channels),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True)
    )
    print(f"EdgeConv initialized with in_channels={in_channels}, out_channels={out_channels}.")

  def forward(self,x,edge_index):
    row,col=edge_index
    x_i=x[row]
    x_j=x[col]
    edge_features=torch.cat([x_i,x_j-x_i],dim=1)
    transformed_features=self.mlp(edge_features)
    aggregated_features=scatter(transformed_features,col,dim=0,reduce='max')
    return aggregated_features


import torch_geometric.nn as tgnn

class KNNGraph(nn.Module):
  def __init__(self,k:int):
    super(KNNGraph,self).__init__()
    self.k=k
    print(f"KNNGraph initialized with k={self.k}.")
  def forward(self,x:torch.Tensor)->torch.Tensor:
    # Ensure x is 2D: (num_nodes, num_features) for knn_graph
    if x.dim() == 3: # If x is (batch_size, num_points, num_features)
        # Reshape to (batch_size * num_points, num_features)
        x_flat = x.view(-1, x.shape[-1])
    else:
        x_flat = x

    # knn_graph expects a 2D tensor of node features
    edge_index=tgnn.knn_graph(x_flat, k=self.k, loop=False, flow='source_to_target')
    # print(f"KNNGraph forward pass completed. Generated edge_index shape: {edge_index.shape}.")
    return edge_index

