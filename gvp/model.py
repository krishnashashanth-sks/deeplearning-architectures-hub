import torch.nn as nn
from layers import GVPGNNLayer,GVPAttentionLayer,GVP
import torch

class GVPGNNStack(nn.Module):
  def __init__(self, num_layers, node_in_dims, node_hidden_dims, node_out_dims, msg_dims, graph_out_dims=None, aggr_fn=torch.sum, layer_type='GNN', head_dims=None):
    super(GVPGNNStack, self).__init__()
    self.layers = nn.ModuleList()
    self.layer_type = layer_type

    if layer_type == 'GNN':
      LayerClass = GVPGNNLayer
      LayerArgs = lambda in_d, out_d: (in_d, out_d, msg_dims, aggr_fn)
    elif layer_type == 'Attention':
      if head_dims is None:
        raise ValueError("head_dims must be provided for 'Attention' layer_type.")
      LayerClass = GVPAttentionLayer
      LayerArgs = lambda in_d, out_d: (in_d, out_d, head_dims)
    else:
      raise ValueError(f"Unknown layer_type: {layer_type}. Choose 'GNN' or 'Attention'.")

    # First layer (input to hidden)
    self.layers.append(LayerClass(*LayerArgs(node_in_dims, node_hidden_dims)))

    # Hidden layers
    for _ in range(num_layers - 2):
      self.layers.append(LayerClass(*LayerArgs(node_hidden_dims, node_hidden_dims)))

    # Last GNN layer (hidden to node_out_dims)
    if num_layers > 1:
      self.layers.append(LayerClass(*LayerArgs(node_hidden_dims, node_out_dims)))
    elif num_layers == 1:
      # If only one layer, it goes directly from input to node_out_dims
      self.layers = nn.ModuleList([LayerClass(*LayerArgs(node_in_dims, node_out_dims))])
    else:
      raise ValueError("num_layers must be at least 1")

    self.graph_out_dims = graph_out_dims
    if graph_out_dims is not None:
      # Readout layer: Global mean pooling followed by a GVP layer
      self.readout_gvp = GVP(node_out_dims, graph_out_dims)

  def forward(self, x, edge_index):
    s, v = x
    for layer in self.layers:
      s, v = layer((s, v), edge_index)

    if self.graph_out_dims is not None:
      # Global mean pooling for readout
      s_pooled = torch.mean(s, dim=0, keepdim=True) # Output shape (1, S_out)
      v_pooled = torch.mean(v, dim=0, keepdim=True) # Output shape (1, V_out, 3)
      s, v = self.readout_gvp((s_pooled, v_pooled))
      # Squeeze the batch dimension if it's 1, to get (S_out) and (V_out, 3)
      # if the downstream task expects single graph output without batch dim
      s = s.squeeze(0)
      v = v.squeeze(0)

    return s, v