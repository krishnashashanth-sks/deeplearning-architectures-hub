from torch_geometric.nn import GATConv
import torch
import torch.nn.functional as F

class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, heads=8):
        super(GAT, self).__init__()
        # GATConv applies attention mechanism
        # `heads` specify the number of attention heads
        # `dropout` is applied to the attention coefficients
        self.conv1 = GATConv(num_node_features, 8, heads=heads, dropout=0.6)
        # On the output layer, we usually collapse the heads back to a single feature vector
        self.conv2 = GATConv(8 * heads, num_classes, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x) # Exponential Linear Unit activation
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)