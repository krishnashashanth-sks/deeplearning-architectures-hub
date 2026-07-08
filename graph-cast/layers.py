import torch.nn as nn
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

# ---  GraphCast Model Definition Functions ---

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, coord_dim=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + coord_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, x, pos):
        x_and_pos = torch.cat([x, pos], dim=-1)
        return self.mlp(x_and_pos)

class MeshGraphProcessorBlock(MessagePassing):
    def __init__(self, latent_dim, edge_dim=0):
        super().__init__(aggr='add')
        self.latent_dim = latent_dim
        self.edge_dim = edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * latent_dim + edge_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x, edge_index, edge_attr=None):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        if edge_attr is not None:
            tmp = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            tmp = torch.cat([x_i, x_j], dim=-1)
        return self.edge_mlp(tmp)

    def update(self, aggr_out, x):
        tmp = torch.cat([x, aggr_out], dim=-1)
        out = self.norm(x + self.node_mlp(tmp))
        return out

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

