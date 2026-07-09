import torch.nn as nn
from layers import EdgeConv,KNNGraph
import torch

class DGCNN(nn.Module):
  def __init__(self,in_channels,num_classes,k=20,embed_dim=1024,drop_prob=0.5):
    super(DGCNN,self).__init__()
    self.k=k
    self.knn_graph_layer=KNNGraph(k=self.k)
    self.conv1=EdgeConv(in_channels,64)
    self.conv2=EdgeConv(64,64)
    self.conv3=EdgeConv(64,128)
    self.final_mlp_before_pool=nn.Sequential(
        nn.Linear(128,embed_dim),
        nn.BatchNorm1d(embed_dim),
        nn.ReLU(inplace=True)
    )
    self.classifier=nn.Sequential(
        nn.Linear(embed_dim,512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=drop_prob),
        nn.Linear(512,256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=drop_prob),
        nn.Linear(256,num_classes)
    )
    print(f"DGCNN initialized with k={k}, in_channels={in_channels}, num_classes={num_classes}, embed_dim={embed_dim}.")

  def forward(self,x):
    # x is expected to be (batch_size, num_points, in_channels)
    # For graph operations, we need (num_nodes, num_features) where num_nodes = batch_size * num_points

    batch_size, num_points, _ = x.shape

    # Reshape for graph processing: (batch_size * num_points, in_channels)
    x_reshaped = x.view(-1, x.shape[-1])

    # First EdgeConv block
    edge_index1 = self.knn_graph_layer(x_reshaped)
    x1 = self.conv1(x_reshaped, edge_index1)

    # Second EdgeConv block
    edge_index2 = self.knn_graph_layer(x1)
    x2 = self.conv2(x1, edge_index2)

    # Third EdgeConv block
    edge_index3 = self.knn_graph_layer(x2)
    x3 = self.conv3(x2, edge_index3)

    # Apply final MLP before global pooling
    x_final = self.final_mlp_before_pool(x3)

    # Global Max Pooling
    # The `x_final` is (batch_size * num_points, embed_dim)
    # We need to reshape it back to (batch_size, num_points, embed_dim) for pooling per batch
    x_pooled = x_final.view(batch_size, num_points, -1)

    # Apply max pooling along the 'num_points' dimension
    # Output will be (batch_size, embed_dim)
    global_features = torch.max(x_pooled, dim=1)[0]

    # Classification head
    logits = self.classifier(global_features)

    return logits