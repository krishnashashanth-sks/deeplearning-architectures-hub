import torch.nn as nn
from layers import Encoder,MeshGraphProcessorBlock,Decoder

class GraphCastModel(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, coord_dim, num_message_passing_steps, edge_dim=0):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim, coord_dim)
        self.processor = nn.ModuleList([
            MeshGraphProcessorBlock(latent_dim, edge_dim) for _ in range(num_message_passing_steps)
        ])
        self.decoder = Decoder(latent_dim, output_dim)

    def forward(self, data):
        x, edge_index, pos = data.x, data.edge_index, data.pos
        x = self.encoder(x, pos)
        for block in self.processor:
            x = block(x, edge_index)
        output = self.decoder(x)
        return output