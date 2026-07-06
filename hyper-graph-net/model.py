import torch.nn as nn
from layers import HypergraphConv
import torch.nn.functional as F

class HGNN(nn.Module):
  def __init__(self,in_channels,hidden_channels,out_channels,num_nodes,num_hyperedges):
    super(HGNN,self).__init__()
    self.conv1=HypergraphConv(in_channels,hidden_channels)
    self.conv2=HypergraphConv(hidden_channels,out_channels) # Changed from HyperedgeConv to HypergraphConv
    self.num_nodes=num_nodes
    self.num_hyperedges=num_hyperedges
  def forward(self,x,node_to_hyperedge_index,hyperedge_to_node_index):
    # First layer
    x = self.conv1(x, node_to_hyperedge_index, hyperedge_to_node_index, self.num_nodes, self.num_hyperedges) # Changed self.conv to self.conv1
    x = F.relu(x)
    x = F.dropout(x, p=0.5, training=self.training) # Corrected F.dropout arguments

    # Second layer
    x = self.conv2(x, node_to_hyperedge_index, hyperedge_to_node_index, self.num_nodes, self.num_hyperedges)
    return F.log_softmax(x,dim=1)