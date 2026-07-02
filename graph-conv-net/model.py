from torch_geometrics import GCNConv
import torch
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16) # First GCN layer: input_features -> 16 output features
        self.conv2 = GCNConv(16, num_classes)      # Second GCN layer: 16 input features -> num_classes output features

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x) # Apply ReLU activation
        x = F.dropout(x, training=self.training) # Apply dropout for regularization
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1) # Apply log_softmax for multi-class classification

