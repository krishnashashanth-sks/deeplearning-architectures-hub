import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

class HypergraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels, **kwargs):
        # Explicitly set node_dim=0 to avoid the IndexError
        super(HypergraphConv, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.lin_node_to_hyperedge = nn.Linear(in_channels, out_channels)
        self.lin_hyperedge_to_node = nn.Linear(out_channels, out_channels)

    def message(self, x_j, norm_j):
        return x_j * norm_j.view(-1, 1)

    def forward(self, x, node_to_hyperedge_index, hyperedge_to_node_index, num_nodes, num_hyperedges):
        # --- Step 1: Nodes -> Hyperedges ---
        x_node_transformed = self.lin_node_to_hyperedge(x) # [num_nodes, out_channels]

        node_deg = degree(node_to_hyperedge_index[0], num_nodes=num_nodes, dtype=x.dtype)
        node_deg_inv_sqrt = node_deg.pow(-0.5)
        node_deg_inv_sqrt.masked_fill_(node_deg_inv_sqrt == float('inf'), 0)

        # We define x as a tuple (source, target) for bipartite mapping
        # x_j will be nodes (source), x_i will be hyperedges (target)
        hyperedge_features = self.propagate(
            node_to_hyperedge_index,
            x=(x_node_transformed, None),
            norm=node_deg_inv_sqrt,
            size=(num_nodes, num_hyperedges)
        )
        hyperedge_features = F.relu(hyperedge_features)

        # --- Step 2: Hyperedges -> Nodes ---
        hyperedge_deg = degree(node_to_hyperedge_index[1], num_nodes=num_hyperedges, dtype=x.dtype)
        hyperedge_deg_inv_sqrt = hyperedge_deg.pow(-0.5)
        hyperedge_deg_inv_sqrt.masked_fill_(hyperedge_deg_inv_sqrt == float('inf'), 0)

        hyperedge_features_transformed = self.lin_hyperedge_to_node(hyperedge_features)

        # x_j will be hyperedges (source), x_i will be nodes (target)
        node_output = self.propagate(
            hyperedge_to_node_index,
            x=(hyperedge_features_transformed, None),
            norm=hyperedge_deg_inv_sqrt,
            size=(num_hyperedges, num_nodes)
        )

        # Final normalization by node degree
        node_output = node_output * node_deg_inv_sqrt.view(-1, 1)
        return node_output