import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,global_mean_pool
from torch_geometric.data import Data

class TopoGNN(nn.Module):
  def __init__(self,num_node_features,num_topo_features,hidden_channels,num_classes):
    super(TopoGNN,self).__init__()
    torch.manual_seed(12345)
    self.num_topo_features = num_topo_features # <--- This line was added
    self.input_dim_gnn=num_node_features+num_topo_features
    self.conv1=GCNConv(self.input_dim_gnn,hidden_channels)
    self.conv2=GCNConv(hidden_channels,hidden_channels)
    self.lin1=nn.Linear(hidden_channels,hidden_channels//2)
    self.lin2=nn.Linear(hidden_channels//2,num_classes)

  def forward(self,x,edge_index,x_topo,batch):
    # Calculate the actual number of graphs in the batch based on the 'batch' tensor
    num_graphs_in_batch_from_batch_tensor = batch.max().item() + 1 if batch.numel() > 0 else 1

    # Case 1: x_topo is 1D (e.g., for batch_size=1, and DataLoader didn't unsqueeze)
    if x_topo.ndim == 1:
        x_topo = x_topo.unsqueeze(0) # Makes it (1, self.num_topo_features)

    # Case 2: x_topo is 2D but has one batch dimension and is flattened (the problematic case)
    # e.g., (1, num_graphs_in_batch * self.num_topo_features)
    # This occurs if DataLoader concatenates all graph's x_topo and wraps it in a single batch dim.
    if x_topo.ndim == 2 and x_topo.shape[0] == 1 and \
       x_topo.shape[1] == (num_graphs_in_batch_from_batch_tensor * self.num_topo_features):
        x_topo = x_topo.view(num_graphs_in_batch_from_batch_tensor, self.num_topo_features)

    # Case 3: x_topo is 2D with batch_size=1, but the actual batch has multiple graphs.
    # This might happen if `follow_batch` failed but didn't flatten the features.
    elif x_topo.ndim == 2 and x_topo.shape[0] == 1 and num_graphs_in_batch_from_batch_tensor > 1:
        x_topo = x_topo.repeat(num_graphs_in_batch_from_batch_tensor, 1)

    # At this point, x_topo should reliably be (num_graphs_in_batch, self.num_topo_features)
    x_topo_broadcast = x_topo[batch]

    # Concatenate original node features with broadcasted topological features
    x_aug=torch.cat([x,x_topo_broadcast],dim=1)

    # Pass through GNN layers
    x=self.conv1(x_aug,edge_index)
    x=F.relu(x)
    x=self.conv2(x,edge_index)
    x=F.relu(x)

    # Global pooling to get a graph-level representation
    x=global_mean_pool(x,batch)

    # Prediction head
    x=self.lin1(x)
    x=F.relu(x)
    return self.lin2(x)